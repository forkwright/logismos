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
        if layer >= self.plan.geometry.layers {
            return PagedLayerOutOfRangeSnafu {
                layer,
                layers: self.plan.geometry.layers,
            }
            .fail();
        }
        Ok(PagedLayerKv {
            pool: self,
            layer,
            tokens: self.committed_tokens,
        })
    }
    /// Preflight whole-call capacity, then stage unpublished page changes.
    pub fn begin_append(&mut self, append_tokens: usize) -> Result<PagedAppend<'_>> {
        if append_tokens == 0 {
            return PagedEmptyAppendSnafu.fail();
        }
        let original_tokens = self.committed_tokens;
        let target = original_tokens
            .checked_add(append_tokens)
            .filter(|x| *x <= self.plan.geometry.max_context)
            .ok_or_else(|| {
                PagedContextOverflowSnafu {
                    committed_tokens: original_tokens,
                    append_tokens,
                    max_context: self.plan.geometry.max_context,
                }
                .build()
            })?;
        let old_pages = self.table.len();
        let target_pages = ceil(target, self.plan.page_tokens.count(), "append target pages")?;
        let partial = original_tokens % self.plan.page_tokens.count() != 0;
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
            original_tokens,
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
    use std::collections::BTreeSet;

    use super::*;
    use crate::Error;

    const LAYERS: usize = 2;
    const ROW_WIDTH: usize = 3;

    #[derive(Clone, Debug, PartialEq)]
    struct ModelRow {
        key: [f32; ROW_WIDTH],
        value: [f32; ROW_WIDTH],
    }

    fn geometry(max_context: usize) -> PagedKvGeometry {
        PagedKvGeometry {
            layers: LAYERS,
            row_width: ROW_WIDTH,
            max_context,
        }
    }

    fn token_seed(token: usize) -> f32 {
        let mut seed = 0.25;
        for _ in 0..token {
            seed += 1.0;
        }
        seed
    }

    fn layer_seed(layer: usize) -> f32 {
        match layer {
            0 => 10.0,
            1 => 100.0,
            _ => 1_000.0,
        }
    }

    fn model_row(layer: usize, token: usize) -> ModelRow {
        let layer = layer_seed(layer);
        let token = token_seed(token);
        ModelRow {
            key: [layer + token, layer - token, token * 2.0],
            value: [layer * 2.0 + token, token - layer, layer + token * 3.0],
        }
    }

    fn empty_model() -> [Vec<ModelRow>; LAYERS] {
        std::array::from_fn(|_| Vec::new())
    }

    fn append_model(model: &mut [Vec<ModelRow>; LAYERS], start: usize, tokens: usize) {
        for layer in 0..LAYERS {
            for relative in 0..tokens {
                model[layer].push(model_row(layer, start + relative));
            }
        }
    }

    fn write_layer(
        txn: &mut PagedAppend<'_>,
        layer: usize,
        start: usize,
        tokens: usize,
    ) -> Result<()> {
        for relative in 0..tokens {
            let row = model_row(layer, start + relative);
            txn.write_layer_row(layer, relative, &row.key, &row.value)?;
        }
        Ok(())
    }

    fn write_all(txn: &mut PagedAppend<'_>, start: usize, tokens: usize) -> Result<()> {
        write_layer(txn, 0, start, tokens)?;
        write_layer(txn, 1, start, tokens)
    }

    fn assert_inventory(pool: &PagedKvPool, held_old_tail: Option<usize>) {
        assert_eq!(pool.table.len(), pool.fills.len());
        let mut seen = BTreeSet::new();
        for bundle in &pool.table {
            assert!(*bundle < pool.plan.bundle_count());
            assert!(seen.insert(*bundle));
        }
        for bundle in &pool.free {
            assert!(*bundle < pool.plan.bundle_count());
            assert!(seen.insert(*bundle));
        }
        if let Some(bundle) = held_old_tail {
            assert!(bundle < pool.plan.bundle_count());
            assert!(seen.insert(bundle));
        }
        assert_eq!(seen.len(), pool.plan.bundle_count());
        assert_eq!(
            pool.table.len() + pool.free.len() + usize::from(held_old_tail.is_some()),
            pool.plan.bundle_count()
        );
    }

    fn assert_future_rows_refused(pool: &PagedKvPool, visible: usize) -> Result<()> {
        for layer in 0..LAYERS {
            let rows = pool.layer_kv(layer)?;
            assert_eq!(rows.tokens(), visible);
            assert!(matches!(
                rows.key_row(visible),
                Err(Error::PagedReadBeyondVisible { .. })
            ));
            assert!(matches!(
                rows.value_row(visible),
                Err(Error::PagedReadBeyondVisible { .. })
            ));
        }
        Ok(())
    }

    fn assert_committed_matches(pool: &PagedKvPool, model: &[Vec<ModelRow>; LAYERS]) -> Result<()> {
        let tokens = model[0].len();
        assert!(model.iter().all(|layer| layer.len() == tokens));
        assert_eq!(pool.committed_tokens(), tokens);
        for (layer, expected_rows) in model.iter().enumerate() {
            let rows = pool.committed().layer_kv(layer)?;
            assert_eq!(rows.tokens(), tokens);
            for (token, expected) in expected_rows.iter().enumerate() {
                assert_eq!(rows.key_row(token)?, expected.key);
                assert_eq!(rows.value_row(token)?, expected.value);
            }
        }
        assert_future_rows_refused(pool, tokens)
    }

    fn tail_payload(pool: &PagedKvPool, bundle: usize, fill: usize) -> Result<Vec<Vec<f32>>> {
        let mut payload = Vec::new();
        for layer in 0..LAYERS {
            for value in [false, true] {
                let range = pool.span(bundle, layer, 0, value, fill)?;
                payload.push(pool.storage[range].to_vec());
            }
        }
        Ok(payload)
    }

    fn assert_tail_payload(
        pool: &PagedKvPool,
        bundle: usize,
        fill: usize,
        expected: &[Vec<f32>],
    ) -> Result<()> {
        assert_eq!(tail_payload(pool, bundle, fill)?, expected);
        Ok(())
    }

    fn seed_committed(
        pool: &mut PagedKvPool,
        model: &mut [Vec<ModelRow>; LAYERS],
        tokens: usize,
    ) -> Result<()> {
        let mut txn = pool.begin_append(tokens)?;
        write_all(&mut txn, 0, tokens)?;
        txn.commit()?;
        append_model(model, 0, tokens);
        assert_inventory(pool, None);
        assert_committed_matches(pool, model)
    }

    fn assert_staged_inventory(txn: &PagedAppend<'_>) {
        assert_inventory(
            txn.pool,
            txn.replaced_tail
                .map(|replacement| replacement.original_bundle),
        );
    }

    fn exercise_partial_tail_rollback(page_tokens: usize, plan: PagedKvPlan) -> Result<()> {
        let mut pool = PagedKvPool::new(plan)?;
        let mut model = empty_model();
        let start = page_tokens - 1;
        assert_inventory(&pool, None);

        {
            let mut seed = pool.begin_append(start)?;
            write_layer(&mut seed, 0, 0, start)?;
            assert_eq!(seed.layer_kv(0)?.tokens(), start);
            assert_eq!(seed.layer_kv(1)?.tokens(), 0);
            assert!(matches!(
                seed.layer_kv(1)?.key_row(0),
                Err(Error::PagedReadBeyondVisible { .. })
            ));
            assert_staged_inventory(&seed);
            write_layer(&mut seed, 1, 0, start)?;
            seed.commit()?;
        }
        append_model(&mut model, 0, start);
        assert_inventory(&pool, None);
        assert_committed_matches(&pool, &model)?;

        let original_bundle = pool.table[0];
        let original_fill = pool.fills[0];
        let original_payload = tail_payload(&pool, original_bundle, original_fill)?;
        {
            let mut incomplete = pool.begin_append(2)?;
            assert!(incomplete.replaced_tail.is_some());
            assert_ne!(incomplete.pool.table[0], original_bundle);
            assert_staged_inventory(&incomplete);
            assert_tail_payload(
                incomplete.pool,
                original_bundle,
                original_fill,
                &original_payload,
            )?;

            let first = model_row(0, start);
            incomplete.write_layer_row(0, 0, &first.key, &first.value)?;
            assert_eq!(incomplete.layer_kv(0)?.tokens(), start + 1);
            assert_eq!(incomplete.layer_kv(0)?.key_row(start)?, first.key);
            assert_eq!(incomplete.layer_kv(0)?.value_row(start)?, first.value);
            let second = model_row(0, start + 1);
            assert!(matches!(
                incomplete.write_layer_row(0, 1, &second.key[..2], &second.value),
                Err(Error::PagedRowWidth { .. })
            ));
            assert!(matches!(
                incomplete.write_layer_row(0, 0, &first.key, &first.value),
                Err(Error::PagedWriteOrder { .. })
            ));
            assert_eq!(incomplete.layer_kv(0)?.tokens(), start + 1);
            assert_eq!(incomplete.layer_kv(0)?.key_row(start)?, first.key);
            assert_eq!(incomplete.layer_kv(0)?.value_row(start)?, first.value);
            assert_eq!(incomplete.layer_kv(1)?.tokens(), start);
            assert_staged_inventory(&incomplete);

            assert!(matches!(
                incomplete.layer_kv(1)?.value_row(start),
                Err(Error::PagedReadBeyondVisible { .. })
            ));
            incomplete.write_layer_row(0, 1, &second.key, &second.value)?;
            assert_tail_payload(
                incomplete.pool,
                original_bundle,
                original_fill,
                &original_payload,
            )?;
            assert_staged_inventory(&incomplete);
            assert!(matches!(
                incomplete.commit(),
                Err(Error::PagedIncompleteAppend { .. })
            ));
        }
        assert_eq!(pool.table[0], original_bundle);
        assert_tail_payload(&pool, original_bundle, original_fill, &original_payload)?;
        assert_inventory(&pool, None);
        assert_committed_matches(&pool, &model)?;

        {
            let mut dropped = pool.begin_append(2)?;
            write_layer(&mut dropped, 0, start, 2)?;
            assert_staged_inventory(&dropped);
        }
        assert_eq!(pool.table[0], original_bundle);
        assert_tail_payload(&pool, original_bundle, original_fill, &original_payload)?;
        assert_inventory(&pool, None);
        assert_committed_matches(&pool, &model)?;

        {
            let mut retry = pool.begin_append(2)?;
            write_all(&mut retry, start, 2)?;
            assert_staged_inventory(&retry);
            retry.commit()?;
        }
        append_model(&mut model, start, 2);
        assert_inventory(&pool, None);
        assert_committed_matches(&pool, &model)
    }

    fn exercise_start_and_multi_page_append(
        page_tokens: usize,
        plan: PagedKvPlan,
        start: usize,
        append_tokens: usize,
    ) -> Result<()> {
        let mut pool = PagedKvPool::new(plan)?;
        let mut model = empty_model();
        seed_committed(&mut pool, &mut model, start)?;

        let old_tail = if start % page_tokens == 0 {
            None
        } else {
            let page = pool.table.len() - 1;
            let bundle = pool.table[page];
            Some((
                bundle,
                pool.fills[page],
                tail_payload(&pool, bundle, pool.fills[page])?,
            ))
        };
        {
            let mut txn = pool.begin_append(append_tokens)?;
            assert_staged_inventory(&txn);
            if let Some((bundle, fill, payload)) = &old_tail {
                assert_ne!(txn.pool.table[txn.pool.table.len() - 2], *bundle);
                assert_tail_payload(txn.pool, *bundle, *fill, payload)?;
            } else {
                assert!(txn.replaced_tail.is_none());
            }
            write_layer(&mut txn, 0, start, append_tokens)?;
            assert_eq!(txn.layer_kv(0)?.tokens(), start + append_tokens);
            assert_eq!(txn.layer_kv(1)?.tokens(), start);
            assert!(matches!(
                txn.layer_kv(1)?.key_row(start),
                Err(Error::PagedReadBeyondVisible { .. })
            ));
            assert_staged_inventory(&txn);
            write_layer(&mut txn, 1, start, append_tokens)?;
            txn.commit()?;
        }
        append_model(&mut model, start, append_tokens);
        assert_inventory(&pool, None);
        assert_committed_matches(&pool, &model)
    }

    #[test]
    fn all_page_sizes_match_independent_transaction_row_models() -> Result<()> {
        for (page_tokens, make_plan) in [
            (
                8,
                PagedKvPlan::b8 as fn(PagedKvGeometry) -> Result<PagedKvPlan>,
            ),
            (
                16,
                PagedKvPlan::b16 as fn(PagedKvGeometry) -> Result<PagedKvPlan>,
            ),
            (
                32,
                PagedKvPlan::b32 as fn(PagedKvGeometry) -> Result<PagedKvPlan>,
            ),
        ] {
            let max_context = page_tokens * 2 + 1;
            exercise_partial_tail_rollback(page_tokens, make_plan(geometry(max_context))?)?;
            exercise_start_and_multi_page_append(
                page_tokens,
                make_plan(geometry(max_context))?,
                page_tokens,
                page_tokens + 1,
            )?;
            exercise_start_and_multi_page_append(
                page_tokens,
                make_plan(geometry(max_context))?,
                page_tokens + 1,
                page_tokens,
            )?;
        }
        Ok(())
    }
    #[test]
    fn boundary_capacity_and_shape_are_typed() -> Result<()> {
        let mut pool = PagedKvPool::new(PagedKvPlan::b8(geometry(20))?)?;
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
        let word = size_of::<usize>();
        let narrow = PagedKvGeometry {
            layers: 1,
            row_width: 1,
            max_context: 128,
        };
        let narrow_b8 = PagedKvPlan::b8(narrow)?;
        let narrow_b16 = PagedKvPlan::b16(narrow)?;
        let narrow_b32 = PagedKvPlan::b32(narrow)?;
        assert_eq!(narrow_b8.backing_bytes(), 1_088);
        assert_eq!(narrow_b8.metadata_bytes(), 50 * word);
        assert_eq!(narrow_b8.total_requested_bytes(), 1_088 + 50 * word);
        assert_eq!(narrow_b16.backing_bytes(), 1_152);
        assert_eq!(narrow_b16.metadata_bytes(), 26 * word);
        assert_eq!(narrow_b16.total_requested_bytes(), 1_152 + 26 * word);
        assert_eq!(narrow_b32.backing_bytes(), 1_280);
        assert_eq!(narrow_b32.metadata_bytes(), 14 * word);
        assert_eq!(narrow_b32.total_requested_bytes(), 1_280 + 14 * word);
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
        assert_eq!(wide_b8.metadata_bytes(), 50 * word);
        assert_eq!(wide_b8.total_requested_bytes(), 69_632 + 50 * word);
        assert_eq!(wide_b16.backing_bytes(), 73_728);
        assert_eq!(wide_b16.metadata_bytes(), 26 * word);
        assert_eq!(wide_b16.total_requested_bytes(), 73_728 + 26 * word);
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
