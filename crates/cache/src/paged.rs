//! Private transactional CPU paged KV storage for one decoder execution.

use core::ops::Range;

use snafu::ResultExt;

use crate::error::{
    PagedAllocationSnafu, PagedAppendTokenOutOfRangeSnafu, PagedArithmeticSnafu,
    PagedCapacitySnafu, PagedContextOverflowSnafu, PagedEmptyAppendSnafu,
    PagedIncompleteAppendSnafu, PagedLayerOutOfRangeSnafu, PagedLayoutSnafu,
    PagedReadBeyondVisibleSnafu, PagedRowWidthSnafu, PagedWriteOrderSnafu, PagedZeroDimensionSnafu,
    Result,
};

/// Geometry shared by an execution plan and its private KV allocation.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvGeometry {
    /// Full-attention transformer layers represented by this pool.
    pub layers: usize,
    /// F32 values in one key or value token row.
    pub row_width: usize,
    /// Maximum retained token count.
    pub max_context: usize,
}

/// Supported token counts in one all-layer page bundle.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum PagedKvPageTokens {
    /// Eight token rows per all-layer page bundle.
    B8,
    /// Sixteen token rows per all-layer page bundle.
    B16,
    /// Thirty-two token rows per all-layer page bundle.
    B32,
}

impl PagedKvPageTokens {
    /// Number of token rows held by one page.
    #[must_use]
    pub const fn count(self) -> usize {
        match self {
            Self::B8 => 8,
            Self::B16 => 16,
            Self::B32 => 32,
        }
    }
}

/// Checked fixed allocation plan for a private paged-KV pool.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvPlan {
    geometry: PagedKvGeometry,
    page_tokens: PagedKvPageTokens,
    page_count: usize,
    bundle_count: usize,
    requested_f32: usize,
    backing_bytes: usize,
    metadata_bytes: usize,
    tail_copy_f32: usize,
    tail_copy_bytes: usize,
    total_requested_bytes: usize,
}

impl PagedKvPlan {
    /// Select the least checked CPU allocation request, then tail-copy cost.
    pub fn select(geometry: PagedKvGeometry) -> Result<Self> {
        let candidates = [
            Self::b8(geometry)?,
            Self::b16(geometry)?,
            Self::b32(geometry)?,
        ];
        let mut selected = candidates[0];
        for candidate in candidates.into_iter().skip(1) {
            if candidate.cost() < selected.cost() {
                selected = candidate;
            }
        }
        Ok(selected)
    }
    /// Construct the eight-token candidate.
    pub fn b8(geometry: PagedKvGeometry) -> Result<Self> {
        Self::new(geometry, PagedKvPageTokens::B8)
    }
    /// Construct the sixteen-token candidate.
    pub fn b16(geometry: PagedKvGeometry) -> Result<Self> {
        Self::new(geometry, PagedKvPageTokens::B16)
    }
    /// Construct the thirty-two-token candidate.
    pub fn b32(geometry: PagedKvGeometry) -> Result<Self> {
        Self::new(geometry, PagedKvPageTokens::B32)
    }
    fn new(geometry: PagedKvGeometry, page_tokens: PagedKvPageTokens) -> Result<Self> {
        for (field, value) in [
            ("layers", geometry.layers),
            ("row_width", geometry.row_width),
            ("max_context", geometry.max_context),
        ] {
            if value == 0 {
                return PagedZeroDimensionSnafu { field }.fail();
            }
        }
        let page_count = ceil(geometry.max_context, page_tokens.count(), "page count")?;
        let bundle_count = page_count.checked_add(1).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "tail spare bundle count",
            }
            .build()
        })?;
        let per_bundle = geometry
            .layers
            .checked_mul(2)
            .and_then(|x| x.checked_mul(page_tokens.count()))
            .and_then(|x| x.checked_mul(geometry.row_width))
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "bundle f32 capacity",
                }
                .build()
            })?;
        let requested_f32 = bundle_count.checked_mul(per_bundle).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "pool f32 capacity",
            }
            .build()
        })?;
        let backing_bytes = requested_f32.checked_mul(size_of::<f32>()).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "pool backing bytes",
            }
            .build()
        })?;
        let metadata_entries = page_count
            .checked_mul(2)
            .and_then(|x| x.checked_add(bundle_count))
            .and_then(|x| x.checked_add(geometry.layers))
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "pool metadata entries",
                }
                .build()
            })?;
        let metadata_bytes = metadata_entries
            .checked_mul(size_of::<usize>())
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "pool metadata bytes",
                }
                .build()
            })?;
        let tail_copy_f32 = geometry
            .layers
            .checked_mul(2)
            .and_then(|x| x.checked_mul(geometry.row_width))
            .and_then(|x| x.checked_mul(page_tokens.count() - 1))
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "COW tail copy capacity",
                }
                .build()
            })?;
        let tail_copy_bytes = tail_copy_f32.checked_mul(size_of::<f32>()).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "COW tail copy bytes",
            }
            .build()
        })?;
        let total_requested_bytes = backing_bytes.checked_add(metadata_bytes).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "total pool requested bytes",
            }
            .build()
        })?;
        Ok(Self {
            geometry,
            page_tokens,
            page_count,
            bundle_count,
            requested_f32,
            backing_bytes,
            metadata_bytes,
            tail_copy_f32,
            tail_copy_bytes,
            total_requested_bytes,
        })
    }
    /// Geometry used to derive this plan.
    #[must_use]
    pub const fn geometry(self) -> PagedKvGeometry {
        self.geometry
    }
    /// Selected page-token witness.
    #[must_use]
    pub const fn page_tokens(self) -> PagedKvPageTokens {
        self.page_tokens
    }
    /// Context-covering page-table capacity.
    #[must_use]
    pub const fn page_count(self) -> usize {
        self.page_count
    }
    /// All-layer bundles, including one COW spare.
    #[must_use]
    pub const fn bundle_count(self) -> usize {
        self.bundle_count
    }
    /// Exact f32 backing, including padding and spare.
    #[must_use]
    pub const fn requested_f32_elements(self) -> usize {
        self.requested_f32
    }
    /// F32 backing expressed in checked bytes before allocation.
    #[must_use]
    pub const fn backing_bytes(self) -> usize {
        self.backing_bytes
    }
    /// Persistent non-f32 metadata capacity.
    #[must_use]
    pub const fn metadata_bytes(self) -> usize {
        self.metadata_bytes
    }
    /// Maximum copied f32 payload for a partial-tail COW.
    #[must_use]
    pub const fn max_tail_copy_f32_elements(self) -> usize {
        self.tail_copy_f32
    }
    /// Maximum partial-tail COW payload expressed in checked bytes.
    #[must_use]
    pub const fn max_tail_copy_bytes(self) -> usize {
        self.tail_copy_bytes
    }
    /// Exact CPU allocation request: fixed backing plus persistent metadata.
    #[must_use]
    pub const fn total_requested_bytes(self) -> usize {
        self.total_requested_bytes
    }
    fn cost(self) -> (usize, usize, usize) {
        (
            self.total_requested_bytes,
            self.tail_copy_bytes,
            self.page_tokens.count(),
        )
    }
    fn bundle_elements(self) -> Result<usize> {
        self.geometry
            .layers
            .checked_mul(2)
            .and_then(|x| x.checked_mul(self.page_tokens.count()))
            .and_then(|x| x.checked_mul(self.geometry.row_width))
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "bundle addressing",
                }
                .build()
            })
    }
}

/// One private committed sequence with fixed all-layer page bundles.
#[derive(Debug)]
pub struct PagedKvPool {
    plan: PagedKvPlan,
    storage: Vec<f32>,
    table: Vec<usize>,
    fills: Vec<usize>,
    free: Vec<usize>,
    staged_rows: Vec<usize>,
    committed_tokens: usize,
}

impl PagedKvPool {
    /// Allocate every f32 and persistent metadata capacity up front.
    pub fn new(plan: PagedKvPlan) -> Result<Self> {
        let mut storage = reserve(plan.requested_f32, "paged-KV f32 backing")?;
        storage.resize(plan.requested_f32, 0.0);
        let table = reserve(plan.page_count, "paged-KV page table")?;
        let fills = reserve(plan.page_count, "paged-KV page fills")?;
        let mut free = reserve(plan.bundle_count, "paged-KV free bundles")?;
        let mut staged_rows = reserve(plan.geometry.layers, "paged-KV transaction rows")?;
        staged_rows.resize(plan.geometry.layers, 0);
        for bundle in (0..plan.bundle_count).rev() {
            free.push(bundle);
        }
        Ok(Self {
            plan,
            storage,
            table,
            fills,
            free,
            staged_rows,
            committed_tokens: 0,
        })
    }
    /// The exact plan used by this owner.
    #[must_use]
    pub const fn plan(&self) -> PagedKvPlan {
        self.plan
    }
    /// Rows committed in every layer.
    #[must_use]
    pub const fn committed_tokens(&self) -> usize {
        self.committed_tokens
    }
    /// Immutable committed view.
    #[must_use]
    pub fn committed(&self) -> PagedKvView<'_> {
        PagedKvView {
            pool: self,
            tokens: self.committed_tokens,
        }
    }
    /// One immutable committed layer.
    pub fn layer_kv(&self, layer: usize) -> Result<PagedLayerKv<'_>> {
        self.committed().layer_kv(layer)
    }
    /// Preflight whole-call capacity, then stage unpublished page changes.
    pub fn begin_append(&mut self, append_tokens: usize) -> Result<PagedAppend<'_>> {
        if append_tokens == 0 {
            return PagedEmptyAppendSnafu.fail();
        }
        let target = self
            .committed_tokens
            .checked_add(append_tokens)
            .filter(|x| *x <= self.plan.geometry.max_context)
            .ok_or_else(|| {
                PagedContextOverflowSnafu {
                    committed_tokens: self.committed_tokens,
                    append_tokens,
                    max_context: self.plan.geometry.max_context,
                }
                .build()
            })?;
        let old_pages = self.table.len();
        let target_pages = ceil(target, self.plan.page_tokens.count(), "append target pages")?;
        let partial = self.committed_tokens % self.plan.page_tokens.count() != 0;
        let new_pages = target_pages.checked_sub(old_pages).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "page table monotonicity",
            }
            .build()
        })?;
        let needed = new_pages.checked_add(usize::from(partial)).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "append spare count",
            }
            .build()
        })?;
        if needed > self.free.len() {
            return PagedCapacitySnafu {
                required_bundles: needed,
                available_bundles: self.free.len(),
            }
            .fail();
        }
        self.staged_rows.fill(0);
        let replaced_tail = if partial {
            Some(self.copy_tail()?)
        } else {
            None
        };
        for _ in 0..new_pages {
            let bundle = self.take_free()?;
            self.table.push(bundle);
            self.fills.push(0);
        }
        Ok(PagedAppend {
            pool: self,
            append_tokens,
            original_tokens: self.committed_tokens,
            original_page_count: old_pages,
            replaced_tail,
            committed: false,
        })
    }
    fn take_free(&mut self) -> Result<usize> {
        self.free.pop().ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "preflighted free bundle",
            }
            .build()
        })
    }
    fn copy_tail(&mut self) -> Result<TailReplacement> {
        let page = self.table.len().checked_sub(1).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "partial tail page",
            }
            .build()
        })?;
        let original_bundle = *self.table.get(page).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "partial tail bundle",
            }
            .build()
        })?;
        let fill = *self.fills.get(page).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "partial tail fill",
            }
            .build()
        })?;
        let replacement_bundle = self.take_free()?;
        let copied = (|| -> Result<()> {
            for layer in 0..self.plan.geometry.layers {
                for value in [false, true] {
                    let source = self.span(original_bundle, layer, 0, value, fill)?;
                    let destination = self.span(replacement_bundle, layer, 0, value, fill)?;
                    self.storage.copy_within(source, destination.start);
                }
            }
            Ok(())
        })();
        if let Err(error) = copied {
            self.free.push(replacement_bundle);
            return Err(error);
        }
        if let Some(bundle) = self.table.get_mut(page) {
            *bundle = replacement_bundle;
        } else {
            return PagedLayoutSnafu {
                operation: "partial tail replacement",
            }
            .fail();
        }
        Ok(TailReplacement {
            page,
            original_bundle,
            fill,
            replacement_bundle,
        })
    }
    fn span(
        &self,
        bundle: usize,
        layer: usize,
        token: usize,
        value: bool,
        rows: usize,
    ) -> Result<Range<usize>> {
        let page_tokens = self.plan.page_tokens.count();
        if bundle >= self.plan.bundle_count
            || layer >= self.plan.geometry.layers
            || token >= page_tokens
            || rows > page_tokens - token
        {
            return PagedLayoutSnafu {
                operation: "page addressing",
            }
            .fail();
        }
        let bundle_start = bundle
            .checked_mul(self.plan.bundle_elements()?)
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "bundle offset",
                }
                .build()
            })?;
        let layer_stride = page_tokens
            .checked_mul(self.plan.geometry.row_width)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "layer stride",
                }
                .build()
            })?;
        let value_offset = if value {
            page_tokens
                .checked_mul(self.plan.geometry.row_width)
                .ok_or_else(|| {
                    PagedArithmeticSnafu {
                        operation: "value offset",
                    }
                    .build()
                })?
        } else {
            0
        };
        let start = layer
            .checked_mul(layer_stride)
            .and_then(|x| bundle_start.checked_add(x))
            .and_then(|x| x.checked_add(value_offset))
            .and_then(|x| {
                token
                    .checked_mul(self.plan.geometry.row_width)
                    .and_then(|y| x.checked_add(y))
            })
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "row offset",
                }
                .build()
            })?;
        let end = rows
            .checked_mul(self.plan.geometry.row_width)
            .and_then(|x| start.checked_add(x))
            .filter(|x| *x <= self.storage.len())
            .ok_or_else(|| {
                PagedLayoutSnafu {
                    operation: "row bounds",
                }
                .build()
            })?;
        Ok(start..end)
    }
}

/// Borrowed append transaction; drop restores all logical committed state.
#[derive(Debug)]
pub struct PagedAppend<'a> {
    pool: &'a mut PagedKvPool,
    append_tokens: usize,
    original_tokens: usize,
    original_page_count: usize,
    replaced_tail: Option<TailReplacement>,
    committed: bool,
}

impl PagedAppend<'_> {
    /// Write one contiguous transaction-relative K/V row for one layer.
    pub fn write_layer_row(
        &mut self,
        layer: usize,
        token: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        self.check_layer(layer)?;
        if token >= self.append_tokens {
            return PagedAppendTokenOutOfRangeSnafu {
                token,
                append_tokens: self.append_tokens,
            }
            .fail();
        }
        let expected = *self.pool.staged_rows.get(layer).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "layer write counter",
            }
            .build()
        })?;
        if token != expected {
            return PagedWriteOrderSnafu {
                layer,
                expected,
                actual: token,
            }
            .fail();
        }
        self.check_row(layer, "key", keys)?;
        self.check_row(layer, "value", values)?;
        let absolute = self.original_tokens.checked_add(token).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "absolute append token",
            }
            .build()
        })?;
        let page_tokens = self.pool.plan.page_tokens.count();
        let page = absolute / page_tokens;
        let within = absolute % page_tokens;
        let bundle = *self.pool.table.get(page).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "append page lookup",
            }
            .build()
        })?;
        let key = self.pool.span(bundle, layer, within, false, 1)?;
        let value = self.pool.span(bundle, layer, within, true, 1)?;
        self.pool
            .storage
            .get_mut(key)
            .ok_or_else(|| {
                PagedLayoutSnafu {
                    operation: "key row write",
                }
                .build()
            })?
            .copy_from_slice(keys);
        self.pool
            .storage
            .get_mut(value)
            .ok_or_else(|| {
                PagedLayoutSnafu {
                    operation: "value row write",
                }
                .build()
            })?
            .copy_from_slice(values);
        if let Some(fill) = self.pool.fills.get_mut(page) {
            *fill = (*fill).max(within + 1);
        } else {
            return PagedLayoutSnafu {
                operation: "append fill",
            }
            .fail();
        }
        if let Some(written) = self.pool.staged_rows.get_mut(layer) {
            *written = written.checked_add(1).ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "layer write count",
                }
                .build()
            })?;
        } else {
            return PagedLayoutSnafu {
                operation: "layer write counter update",
            }
            .fail();
        }
        Ok(())
    }
    /// View only committed rows plus this layer's contiguous written prefix.
    pub fn layer_kv(&self, layer: usize) -> Result<PagedLayerKv<'_>> {
        self.check_layer(layer)?;
        let staged = *self.pool.staged_rows.get(layer).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "staged view counter",
            }
            .build()
        })?;
        let tokens = self.original_tokens.checked_add(staged).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "staged visible tokens",
            }
            .build()
        })?;
        Ok(PagedLayerKv {
            pool: self.pool,
            layer,
            tokens,
        })
    }
    /// Publish only after every layer owns every requested row.
    pub fn commit(mut self) -> Result<()> {
        for (layer, written_tokens) in self.pool.staged_rows.iter().copied().enumerate() {
            if written_tokens != self.append_tokens {
                return PagedIncompleteAppendSnafu {
                    layer,
                    written_tokens,
                    append_tokens: self.append_tokens,
                }
                .fail();
            }
        }
        self.pool.committed_tokens = self
            .original_tokens
            .checked_add(self.append_tokens)
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "commit token count",
                }
                .build()
            })?;
        if let Some(replacement) = self.replaced_tail.take() {
            self.pool.free.push(replacement.original_bundle);
        }
        self.pool.staged_rows.fill(0);
        self.committed = true;
        Ok(())
    }
    fn check_layer(&self, layer: usize) -> Result<()> {
        if layer >= self.pool.plan.geometry.layers {
            return PagedLayerOutOfRangeSnafu {
                layer,
                layers: self.pool.plan.geometry.layers,
            }
            .fail();
        }
        Ok(())
    }
    fn check_row(&self, layer: usize, kind: &'static str, row: &[f32]) -> Result<()> {
        if row.len() != self.pool.plan.geometry.row_width {
            return PagedRowWidthSnafu {
                kind,
                layer,
                actual: row.len(),
                expected: self.pool.plan.geometry.row_width,
            }
            .fail();
        }
        Ok(())
    }
}

impl Drop for PagedAppend<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        while self.pool.table.len() > self.original_page_count {
            if let Some(bundle) = self.pool.table.pop() {
                self.pool.free.push(bundle);
            }
            let _ = self.pool.fills.pop();
        }
        if let Some(replacement) = self.replaced_tail {
            if let Some(bundle) = self.pool.table.get_mut(replacement.page) {
                *bundle = replacement.original_bundle;
            }
            if let Some(fill) = self.pool.fills.get_mut(replacement.page) {
                *fill = replacement.fill;
            }
            self.pool.free.push(replacement.replacement_bundle);
        }
        self.pool.staged_rows.fill(0);
    }
}

/// Immutable committed pool view.
#[derive(Debug)]
pub struct PagedKvView<'a> {
    pool: &'a PagedKvPool,
    tokens: usize,
}
impl PagedKvView<'_> {
    /// Visible committed token count.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
    /// Borrow one committed layer.
    pub fn layer_kv(&self, layer: usize) -> Result<PagedLayerKv<'_>> {
        if layer >= self.pool.plan.geometry.layers {
            return PagedLayerOutOfRangeSnafu {
                layer,
                layers: self.pool.plan.geometry.layers,
            }
            .fail();
        }
        Ok(PagedLayerKv {
            pool: self.pool,
            layer,
            tokens: self.tokens,
        })
    }
}

/// Borrowed K/V rows for one layer.
#[derive(Debug)]
pub struct PagedLayerKv<'a> {
    pool: &'a PagedKvPool,
    layer: usize,
    tokens: usize,
}
impl PagedLayerKv<'_> {
    /// Safely visible rows.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
    /// Borrow one key row.
    pub fn key_row(&self, token: usize) -> Result<&[f32]> {
        self.row(token, false)
    }
    /// Borrow one value row.
    pub fn value_row(&self, token: usize) -> Result<&[f32]> {
        self.row(token, true)
    }
    fn row(&self, token: usize, value: bool) -> Result<&[f32]> {
        if token >= self.tokens {
            return PagedReadBeyondVisibleSnafu {
                token,
                visible_tokens: self.tokens,
            }
            .fail();
        }
        let page_tokens = self.pool.plan.page_tokens.count();
        let page = token / page_tokens;
        let within = token % page_tokens;
        let fill = *self.pool.fills.get(page).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "visible fill lookup",
            }
            .build()
        })?;
        if within >= fill {
            return PagedLayoutSnafu {
                operation: "visible fill validation",
            }
            .fail();
        }
        let bundle = *self.pool.table.get(page).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "visible page lookup",
            }
            .build()
        })?;
        let range = self.pool.span(bundle, self.layer, within, value, 1)?;
        self.pool.storage.get(range).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "visible row backing",
            }
            .build()
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TailReplacement {
    page: usize,
    original_bundle: usize,
    fill: usize,
    replacement_bundle: usize,
}

fn ceil(value: usize, divisor: usize, operation: &'static str) -> Result<usize> {
    value
        .checked_add(divisor - 1)
        .and_then(|x| x.checked_div(divisor))
        .ok_or_else(|| PagedArithmeticSnafu { operation }.build())
}
fn reserve<T>(length: usize, target: &'static str) -> Result<Vec<T>> {
    allocation_layout::<T>(length, target)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .context(PagedAllocationSnafu { target })?;
    Ok(values)
}
fn allocation_layout<T>(length: usize, target: &'static str) -> Result<()> {
    std::alloc::Layout::array::<T>(length)
        .map(|_| ())
        .map_err(|_| PagedLayoutSnafu { operation: target }.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    fn geometry() -> PagedKvGeometry {
        PagedKvGeometry {
            layers: 2,
            row_width: 3,
            max_context: 20,
        }
    }
    fn write_all(txn: &mut PagedAppend<'_>, tokens: usize, value: f32) -> Result<()> {
        for layer in 0..2 {
            for token in 0..tokens {
                txn.write_layer_row(
                    layer,
                    token,
                    &[value + layer as f32; 3],
                    &[value + 10.0 + layer as f32; 3],
                )?;
            }
        }
        Ok(())
    }
    #[test]
    fn supported_pages_commit_exact_rows() -> Result<()> {
        for plan in [
            PagedKvPlan::b8(geometry())?,
            PagedKvPlan::b16(geometry())?,
            PagedKvPlan::b32(geometry())?,
        ] {
            let mut pool = PagedKvPool::new(plan)?;
            let mut txn = pool.begin_append(2)?;
            write_all(&mut txn, 2, 1.0)?;
            txn.commit()?;
            let view = pool.layer_kv(1)?;
            assert_eq!(view.tokens(), 2);
            assert_eq!(view.key_row(1)?, &[2.0; 3]);
            assert_eq!(view.value_row(1)?, &[12.0; 3]);
        }
        Ok(())
    }
    #[test]
    fn partial_tail_drop_restores_and_retry_is_clean() -> Result<()> {
        let mut pool = PagedKvPool::new(PagedKvPlan::b8(geometry())?)?;
        {
            let mut txn = pool.begin_append(3)?;
            write_all(&mut txn, 3, 1.0)?;
            txn.commit()?;
        }
        {
            let mut txn = pool.begin_append(2)?;
            write_all(&mut txn, 2, 20.0)?;
            assert_eq!(txn.layer_kv(0)?.tokens(), 5);
        }
        assert_eq!(pool.committed_tokens(), 3);
        assert_eq!(pool.layer_kv(0)?.key_row(2)?, &[1.0; 3]);
        let mut retry = pool.begin_append(2)?;
        write_all(&mut retry, 2, 30.0)?;
        retry.commit()?;
        assert_eq!(pool.layer_kv(0)?.key_row(3)?, &[30.0; 3]);
        Ok(())
    }
    #[test]
    fn page_crossing_appends_keep_prior_full_rows() -> Result<()> {
        let mut pool = PagedKvPool::new(PagedKvPlan::b8(geometry())?)?;
        let mut first = pool.begin_append(7)?;
        write_all(&mut first, 7, 1.0)?;
        first.commit()?;
        let mut crossing = pool.begin_append(2)?;
        write_all(&mut crossing, 2, 9.0)?;
        crossing.commit()?;
        let layer = pool.layer_kv(0)?;
        assert_eq!(layer.tokens(), 9);
        assert_eq!(layer.key_row(6)?, &[1.0; 3]);
        assert_eq!(layer.key_row(7)?, &[9.0; 3]);
        assert_eq!(layer.key_row(8)?, &[9.0; 3]);
        Ok(())
    }
    #[test]
    fn incomplete_or_out_of_order_rows_never_publish() -> Result<()> {
        let mut pool = PagedKvPool::new(PagedKvPlan::b8(geometry())?)?;
        let mut txn = pool.begin_append(2)?;
        assert!(matches!(
            txn.write_layer_row(0, 1, &[1.0; 3], &[2.0; 3]),
            Err(Error::PagedWriteOrder { .. })
        ));
        txn.write_layer_row(0, 0, &[1.0; 3], &[2.0; 3])?;
        assert!(matches!(
            txn.commit(),
            Err(Error::PagedIncompleteAppend { .. })
        ));
        assert_eq!(pool.committed_tokens(), 0);
        assert!(matches!(
            pool.layer_kv(0)?.key_row(0),
            Err(Error::PagedReadBeyondVisible { .. })
        ));
        Ok(())
    }
    #[test]
    fn boundary_capacity_and_shape_are_typed() -> Result<()> {
        let mut pool = PagedKvPool::new(PagedKvPlan::b8(geometry())?)?;
        assert!(matches!(
            pool.begin_append(21),
            Err(Error::PagedContextOverflow { .. })
        ));
        let mut txn = pool.begin_append(1)?;
        assert!(matches!(
            txn.write_layer_row(0, 0, &[1.0; 2], &[2.0; 3]),
            Err(Error::PagedRowWidth { .. })
        ));
        Ok(())
    }
    #[test]
    fn selector_minimizes_checked_cpu_request_not_page_size() -> Result<()> {
        let narrow = PagedKvGeometry {
            layers: 1,
            row_width: 1,
            max_context: 128,
        };
        let narrow_b8 = PagedKvPlan::b8(narrow)?;
        let narrow_b16 = PagedKvPlan::b16(narrow)?;
        let narrow_b32 = PagedKvPlan::b32(narrow)?;
        assert_eq!(narrow_b8.backing_bytes(), 1_088);
        assert_eq!(narrow_b8.metadata_bytes(), 400);
        assert_eq!(narrow_b8.total_requested_bytes(), 1_488);
        assert_eq!(narrow_b16.backing_bytes(), 1_152);
        assert_eq!(narrow_b16.metadata_bytes(), 208);
        assert_eq!(narrow_b16.total_requested_bytes(), 1_360);
        assert_eq!(narrow_b32.backing_bytes(), 1_280);
        assert_eq!(narrow_b32.metadata_bytes(), 112);
        assert_eq!(narrow_b32.total_requested_bytes(), 1_392);
        assert_eq!(
            PagedKvPlan::select(narrow)?.page_tokens(),
            PagedKvPageTokens::B16
        );
        let wide = PagedKvGeometry {
            layers: 1,
            row_width: 64,
            max_context: 128,
        };
        let wide_b8 = PagedKvPlan::b8(wide)?;
        let wide_b16 = PagedKvPlan::b16(wide)?;
        assert_eq!(wide_b8.backing_bytes(), 69_632);
        assert_eq!(wide_b8.metadata_bytes(), 400);
        assert_eq!(wide_b8.total_requested_bytes(), 70_032);
        assert_eq!(wide_b16.backing_bytes(), 73_728);
        assert_eq!(wide_b16.metadata_bytes(), 208);
        assert_eq!(wide_b16.total_requested_bytes(), 73_936);
        assert_eq!(
            PagedKvPlan::select(wide)?.page_tokens(),
            PagedKvPageTokens::B8
        );
        assert!(matches!(
            PagedKvPlan::b8(PagedKvGeometry {
                layers: usize::MAX,
                row_width: 2,
                max_context: 1
            }),
            Err(Error::PagedArithmetic { .. })
        ));
        Ok(())
    }
}
