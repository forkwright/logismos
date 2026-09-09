//! Private CPU paged-KV owner for one decoder sequence.

use crate::error::{MsgSnafu, Result};

/// Storage geometry owned by one paged cache.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvGeometry {
    /// Transformer layer count.
    pub layers: usize,
    /// F32 values in one K or V token row.
    pub row_width: usize,
    /// Maximum retained token count.
    pub max_context: usize,
}

#[derive(Clone, Copy, Debug)]
enum PageTokens { B8 = 8, B16 = 16, B32 = 32 }

impl PageTokens { const fn count(self) -> usize { self as usize } }

/// Checked allocation plan for the private all-layer page pool.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvPlan { geometry: PagedKvGeometry, page_tokens: PageTokens, bundles: usize, requested_f32: usize, metadata_bytes: usize }

impl PagedKvPlan {
    /// Select the smallest exact requested backing plus bounded metadata; ties use smaller pages.
    pub fn select(geometry: PagedKvGeometry) -> Result<Self> {
        let mut best = None;
        for page in [PageTokens::B8, PageTokens::B16, PageTokens::B32] {
            let plan = Self::with_page(geometry, page)?;
            if best.as_ref().is_none_or(|old: &Self| (plan.requested_f32, plan.metadata_bytes, page.count()) < (old.requested_f32, old.metadata_bytes, old.page_tokens.count())) { best = Some(plan); }
        }
        best.ok_or_else(|| MsgSnafu { message: "paged KV selector has no page candidates".to_string() }.build())
    }
    /// Construct a typed page-size witness plan.
    pub fn b8(geometry: PagedKvGeometry) -> Result<Self> { Self::with_page(geometry, PageTokens::B8) }
    /// Construct a typed page-size witness plan.
    pub fn b16(geometry: PagedKvGeometry) -> Result<Self> { Self::with_page(geometry, PageTokens::B16) }
    /// Construct a typed page-size witness plan.
    pub fn b32(geometry: PagedKvGeometry) -> Result<Self> { Self::with_page(geometry, PageTokens::B32) }
    fn with_page(geometry: PagedKvGeometry, page_tokens: PageTokens) -> Result<Self> {
        if geometry.layers == 0 || geometry.row_width == 0 || geometry.max_context == 0 { return Err(MsgSnafu { message: "paged KV geometry dimensions must be positive".to_string() }.build()); }
        let pages = geometry.max_context.checked_add(page_tokens.count() - 1).and_then(|n| n.checked_div(page_tokens.count())).ok_or_else(|| MsgSnafu { message: "paged KV page count overflows".to_string() }.build())?;
        let bundles = pages.checked_add(1).ok_or_else(|| MsgSnafu { message: "paged KV tail spare overflows".to_string() }.build())?;
        let requested_f32 = bundles.checked_mul(geometry.layers).and_then(|n| n.checked_mul(2)).and_then(|n| n.checked_mul(page_tokens.count())).and_then(|n| n.checked_mul(geometry.row_width)).ok_or_else(|| MsgSnafu { message: "paged KV backing overflows".to_string() }.build())?;
        let metadata_bytes = pages.checked_mul(core::mem::size_of::<usize>() + core::mem::size_of::<u8>()).and_then(|n| n.checked_add(core::mem::size_of::<usize>())).ok_or_else(|| MsgSnafu { message: "paged KV metadata overflows".to_string() }.build())?;
        Ok(Self { geometry, page_tokens, bundles, requested_f32, metadata_bytes })
    }
    /// Exact fixed f32 backing allocation, including the tail-COW spare.
    #[must_use] pub const fn requested_f32_elements(self) -> usize { self.requested_f32 }
}

/// One private committed sequence and its fixed all-layer bundle pool.
pub struct PagedKvPool { plan: PagedKvPlan, storage: Vec<f32>, table: Vec<usize>, fills: Vec<usize>, free: Vec<usize>, committed_tokens: usize }

impl PagedKvPool {
    /// Allocate fixed f32 backing and one private sequence table.
    pub fn new(plan: PagedKvPlan) -> Result<Self> {
        let pages = plan.bundles - 1;
        let mut free: Vec<usize> = (0..plan.bundles).rev().collect();
        let mut table = Vec::with_capacity(pages); let mut fills = Vec::with_capacity(pages);
        for _ in 0..pages { table.push(free.pop().ok_or_else(|| MsgSnafu { message: "paged KV pool cannot initialize table".to_string() }.build())?); fills.push(0); }
        Ok(Self { storage: vec![0.0; plan.requested_f32], plan, table, fills, free, committed_tokens: 0 })
    }
    /// Preflight and stage a contiguous token append.
    pub fn begin_append(&mut self, token_count: usize) -> Result<PagedAppend<'_>> {
        if token_count == 0 || self.committed_tokens.checked_add(token_count).is_none_or(|n| n > self.plan.geometry.max_context) { return Err(MsgSnafu { message: "paged KV append exceeds checked context".to_string() }.build()); }
        let table = self.table.clone(); let fills = self.fills.clone();
        Ok(PagedAppend { pool: self, token_count, table, fills, rows_written: vec![false; self.plan.geometry.layers], committed: false })
    }
    /// Read one committed layer without materializing the cache.
    pub fn layer_kv(&self, layer: usize) -> Result<PagedLayerKv<'_>> { PagedLayerKv::new(self, &self.table, self.committed_tokens, layer) }
}

/// Staged all-layer token append. Drop restores the committed logical table.
pub struct PagedAppend<'a> { pool: &'a mut PagedKvPool, token_count: usize, table: Vec<usize>, fills: Vec<usize>, rows_written: Vec<bool>, committed: bool }

impl PagedAppend<'_> {
    /// Stage one K/V row for a layer; keys and values are whole decoder-owned rows.
    pub fn write_layer_row(&mut self, layer: usize, keys: &[f32], values: &[f32]) -> Result<()> {
        if layer >= self.pool.plan.geometry.layers || keys.len() != self.pool.plan.geometry.row_width || values.len() != keys.len() { return Err(MsgSnafu { message: "paged KV layer row shape is invalid".to_string() }.build()); }
        if self.token_count != 1 { return Err(MsgSnafu { message: "paged KV multi-token rows require an explicit future batch writer".to_string() }.build()); }
        let token = self.pool.committed_tokens; let page = token / self.pool.plan.page_tokens.count(); let slot = token % self.pool.plan.page_tokens.count();
        let bundle = self.table[page]; let width = self.pool.plan.geometry.row_width; let page_rows = self.pool.plan.page_tokens.count(); let layer_stride = 2 * page_rows * width; let start = bundle * self.pool.plan.geometry.layers * layer_stride + layer * layer_stride + slot * width;
        self.pool.storage[start..start + width].copy_from_slice(keys); self.pool.storage[start + page_rows * width..start + page_rows * width + width].copy_from_slice(values); self.fills[page] = slot + 1; self.rows_written[layer] = true; Ok(())
    }
    /// Read staged prior plus current rows for one layer.
    pub fn layer_kv(&self, layer: usize) -> Result<PagedLayerKv<'_>> { PagedLayerKv::new(self.pool, &self.table, self.pool.committed_tokens + self.token_count, layer) }
    /// Publish only after every layer has staged its row.
    pub fn commit(mut self) -> Result<()> { if self.rows_written.iter().any(|written| !written) { return Err(MsgSnafu { message: "paged KV append is incomplete across layers".to_string() }.build()); } self.pool.table = self.table.clone(); self.pool.fills = self.fills.clone(); self.pool.committed_tokens += self.token_count; self.committed = true; Ok(()) }
}

impl Drop for PagedAppend<'_> { fn drop(&mut self) { if !self.committed { /* logical state stayed in pool until commit */ } } }

/// Borrowed whole-row K/V view; decoder owns head subranges.
pub struct PagedLayerKv<'a> { pool: &'a PagedKvPool, table: &'a [usize], tokens: usize, layer: usize }
impl<'a> PagedLayerKv<'a> {
    fn new(pool: &'a PagedKvPool, table: &'a [usize], tokens: usize, layer: usize) -> Result<Self> { if layer >= pool.plan.geometry.layers { return Err(MsgSnafu { message: "paged KV layer out of range".to_string() }.build()); } Ok(Self { pool, table, tokens, layer }) }
    #[must_use] pub const fn tokens(&self) -> usize { self.tokens }
    pub fn key_row(&self, token: usize) -> Result<&'a [f32]> { self.row(token, false) }
    pub fn value_row(&self, token: usize) -> Result<&'a [f32]> { self.row(token, true) }
    fn row(&self, token: usize, value: bool) -> Result<&'a [f32]> { if token >= self.tokens { return Err(MsgSnafu { message: "paged KV token read exceeds visible rows".to_string() }.build()); } let width=self.pool.plan.geometry.row_width; let rows=self.pool.plan.page_tokens.count(); let page=token/rows; let slot=token%rows; let bundle=self.table[page]; let stride=2*rows*width; let start=bundle*self.pool.plan.geometry.layers*stride+self.layer*stride+(if value { rows*width } else { 0 })+slot*width; Ok(&self.pool.storage[start..start+width]) }
}
