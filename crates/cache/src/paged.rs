//! Private transactional CPU paged KV storage for one decoder execution.

use core::ops::Range;

use snafu::ResultExt;

#[cfg(feature = "gpu")]
use hipcore::{Device, DeviceBuffer, Stream};
#[cfg(feature = "gpu")]
use kernels::attention::{NativePageTokens, NativePagedDecodePlan, NativePagedPrefillPlan};
#[cfg(feature = "gpu")]
use kernels::numerical_status::NativeNumericalStatus;

use crate::error::{
    PagedAllocationSnafu, PagedAppendTokenOutOfRangeSnafu, PagedArithmeticSnafu,
    PagedCapacitySnafu, PagedContextOverflowSnafu, PagedEmptyAppendSnafu,
    PagedIncompleteAppendSnafu, PagedLayerOutOfRangeSnafu, PagedLayoutSnafu,
    PagedReadBeyondVisibleSnafu, PagedRowWidthSnafu, PagedWriteOrderSnafu, PagedZeroDimensionSnafu,
    Result,
};
#[cfg(any(feature = "gpu", test))]
use crate::error::{PagedNativeCommitNotPreparedSnafu, PagedNativeCommitPreparedSnafu};
#[cfg(feature = "gpu")]
use crate::error::{PagedNativeDeviceMismatchSnafu, PagedNativePoisonedSnafu};

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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PageTokens {
    B8,
    B16,
    B32,
}

impl PageTokens {
    const fn count(self) -> usize {
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
    page_tokens: PageTokens,
    allocation: PagedKvAllocation,
}

#[derive(Clone, Copy, Debug)]
struct PagedKvAllocation {
    page_count: usize,
    bundle_count: usize,
    requested_f32: usize,
    tail_copy_bytes: usize,
    total_requested_bytes: usize,
}

impl PagedKvPlan {
    /// Select the least checked CPU allocation request, then tail-copy cost.
    pub fn select(geometry: PagedKvGeometry) -> Result<Self> {
        let candidates = [
            Self::new(geometry, PageTokens::B8)?,
            Self::new(geometry, PageTokens::B16)?,
            Self::new(geometry, PageTokens::B32)?,
        ];
        let mut selected = candidates[0];
        for candidate in candidates.into_iter().skip(1) {
            if candidate.cost() < selected.cost() {
                selected = candidate;
            }
        }
        Ok(selected)
    }
    fn new(geometry: PagedKvGeometry, page_tokens: PageTokens) -> Result<Self> {
        Self::validate_geometry(geometry)?;
        let allocation = Self::allocation(geometry, page_tokens)?;
        Ok(Self {
            geometry,
            page_tokens,
            allocation,
        })
    }
    fn validate_geometry(geometry: PagedKvGeometry) -> Result<()> {
        for (field, value) in [
            ("layers", geometry.layers),
            ("row_width", geometry.row_width),
            ("max_context", geometry.max_context),
        ] {
            if value == 0 {
                return PagedZeroDimensionSnafu { field }.fail();
            }
        }
        Ok(())
    }
    fn allocation(geometry: PagedKvGeometry, page_tokens: PageTokens) -> Result<PagedKvAllocation> {
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
        Ok(PagedKvAllocation {
            page_count,
            bundle_count,
            requested_f32,
            tail_copy_bytes,
            total_requested_bytes,
        })
    }
    /// Exact f32 backing, including padding and spare.
    #[must_use]
    pub const fn requested_f32_elements(self) -> usize {
        self.allocation.requested_f32
    }
    fn cost(self) -> (usize, usize, usize) {
        (
            self.allocation.total_requested_bytes,
            self.allocation.tail_copy_bytes,
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

/// The HIP-free logical state shared by CPU and optional native paged backings.
///
/// WHY: page ownership, fills, append ordering, and publication are sequence
/// semantics, while a host `Vec` and device buffers are only physical storage.
#[derive(Debug)]
struct PagedKvLedger {
    plan: PagedKvPlan,
    table: Vec<usize>,
    fills: Vec<usize>,
    free: Vec<usize>,
    staged_rows: Vec<usize>,
    committed_tokens: usize,
}

impl PagedKvLedger {
    fn new(plan: PagedKvPlan) -> Result<Self> {
        let table = reserve(plan.allocation.page_count, "paged-KV page table")?;
        let fills = reserve(plan.allocation.page_count, "paged-KV page fills")?;
        let mut free = reserve(plan.allocation.bundle_count, "paged-KV free bundles")?;
        let mut staged_rows = reserve(plan.geometry.layers, "paged-KV transaction rows")?;
        staged_rows.resize(plan.geometry.layers, 0);
        for bundle in (0..plan.allocation.bundle_count).rev() {
            free.push(bundle);
        }
        Ok(Self {
            plan,
            table,
            fills,
            free,
            staged_rows,
            committed_tokens: 0,
        })
    }

    fn begin_append(&mut self, append_tokens: usize) -> Result<AppendReservation> {
        if append_tokens == 0 {
            return PagedEmptyAppendSnafu.fail();
        }
        let original_tokens = self.committed_tokens;
        let target = original_tokens
            .checked_add(append_tokens)
            .filter(|value| *value <= self.plan.geometry.max_context)
            .ok_or_else(|| {
                PagedContextOverflowSnafu {
                    committed_tokens: original_tokens,
                    append_tokens,
                    max_context: self.plan.geometry.max_context,
                }
                .build()
            })?;
        let original_page_count = self.table.len();
        let page_tokens = self.plan.page_tokens.count();
        let target_pages = ceil(target, page_tokens, "append target pages")?;
        let partial = !original_tokens.is_multiple_of(page_tokens);
        let new_pages = target_pages
            .checked_sub(original_page_count)
            .ok_or_else(|| {
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
            Some(self.replace_tail_without_copy()?)
        } else {
            None
        };
        for _ in 0..new_pages {
            let bundle = self.take_free()?;
            self.table.push(bundle);
            self.fills.push(0);
        }
        Ok(AppendReservation {
            append_tokens,
            original_tokens,
            target_tokens: target,
            original_page_count,
            replaced_tail,
        })
    }

    fn replace_tail_without_copy(&mut self) -> Result<TailReplacement> {
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
        *self.table.get_mut(page).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "partial tail replacement",
            }
            .build()
        })? = replacement_bundle;
        Ok(TailReplacement {
            page,
            original_bundle,
            fill,
            replacement_bundle,
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

    fn validate_commit(&self, reservation: &AppendReservation) -> Result<()> {
        for (layer, written_tokens) in self.staged_rows.iter().copied().enumerate() {
            if written_tokens != reservation.append_tokens {
                return PagedIncompleteAppendSnafu {
                    layer,
                    written_tokens,
                    append_tokens: reservation.append_tokens,
                }
                .fail();
            }
        }
        Ok(())
    }

    fn publish_commit(&mut self, reservation: &mut AppendReservation) {
        self.committed_tokens = reservation.target_tokens;
        if let Some(replacement) = reservation.replaced_tail.take() {
            self.free.push(replacement.original_bundle);
        }
        self.staged_rows.fill(0);
    }

    fn rollback(&mut self, reservation: &AppendReservation) {
        while self.table.len() > reservation.original_page_count {
            if let Some(bundle) = self.table.pop() {
                self.free.push(bundle);
            }
            let _ = self.fills.pop();
        }
        if let Some(replacement) = reservation.replaced_tail {
            if let Some(bundle) = self.table.get_mut(replacement.page) {
                *bundle = replacement.original_bundle;
            }
            if let Some(fill) = self.fills.get_mut(replacement.page) {
                *fill = replacement.fill;
            }
            self.free.push(replacement.replacement_bundle);
        }
        self.staged_rows.fill(0);
    }

    fn write_location(
        &self,
        reservation: &AppendReservation,
        layer: usize,
        token: usize,
    ) -> Result<LedgerWriteLocation> {
        self.check_layer(layer)?;
        if token >= reservation.append_tokens {
            return PagedAppendTokenOutOfRangeSnafu {
                token,
                append_tokens: reservation.append_tokens,
            }
            .fail();
        }
        let expected = *self.staged_rows.get(layer).ok_or_else(|| {
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
        let absolute = reservation
            .original_tokens
            .checked_add(token)
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "absolute append token",
                }
                .build()
            })?;
        let page_tokens = self.plan.page_tokens.count();
        let page = absolute / page_tokens;
        let within = absolute % page_tokens;
        let bundle = *self.table.get(page).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "append page lookup",
            }
            .build()
        })?;
        Ok(LedgerWriteLocation {
            page,
            bundle,
            within,
        })
    }

    fn record_write(&mut self, layer: usize, page: usize, within: usize) -> Result<()> {
        if let Some(fill) = self.fills.get_mut(page) {
            *fill = (*fill).max(within + 1);
        } else {
            return PagedLayoutSnafu {
                operation: "append fill",
            }
            .fail();
        }
        if let Some(written) = self.staged_rows.get_mut(layer) {
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

    fn visible_tokens(&self, reservation: &AppendReservation, layer: usize) -> Result<usize> {
        self.check_layer(layer)?;
        let staged = *self.staged_rows.get(layer).ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "staged view counter",
            }
            .build()
        })?;
        reservation
            .original_tokens
            .checked_add(staged)
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "staged visible tokens",
                }
                .build()
            })
    }

    fn check_layer(&self, layer: usize) -> Result<()> {
        if layer >= self.plan.geometry.layers {
            return PagedLayerOutOfRangeSnafu {
                layer,
                layers: self.plan.geometry.layers,
            }
            .fail();
        }
        Ok(())
    }
}

/// One private committed sequence with fixed all-layer page bundles.
#[derive(Debug)]
pub struct PagedKvPool {
    ledger: PagedKvLedger,
    storage: Vec<f32>,
}

impl PagedKvPool {
    /// Allocate every f32 and persistent metadata capacity up front.
    pub fn new(plan: PagedKvPlan) -> Result<Self> {
        let mut storage = reserve(plan.allocation.requested_f32, "paged-KV f32 backing")?;
        storage.resize(plan.allocation.requested_f32, 0.0);
        Ok(Self {
            ledger: PagedKvLedger::new(plan)?,
            storage,
        })
    }
    /// One immutable committed layer.
    pub fn layer_kv(&self, layer: usize) -> Result<PagedLayerKv<'_>> {
        if layer >= self.ledger.plan.geometry.layers {
            return PagedLayerOutOfRangeSnafu {
                layer,
                layers: self.ledger.plan.geometry.layers,
            }
            .fail();
        }
        Ok(PagedLayerKv {
            pool: self,
            layer,
            tokens: self.ledger.committed_tokens,
        })
    }
    /// Preflight whole-call capacity, then stage unpublished page changes.
    pub fn begin_append(&mut self, append_tokens: usize) -> Result<PagedAppend<'_>> {
        let reservation = self.ledger.begin_append(append_tokens)?;
        if let Some(replacement) = reservation.replaced_tail
            && let Err(error) = self.copy_tail(replacement)
        {
            self.ledger.rollback(&reservation);
            return Err(error);
        }
        Ok(PagedAppend {
            pool: self,
            reservation,
            committed: false,
        })
    }
    fn copy_tail(&mut self, replacement: TailReplacement) -> Result<()> {
        for layer in 0..self.ledger.plan.geometry.layers {
            for value in [false, true] {
                let source = self.span(
                    replacement.original_bundle,
                    layer,
                    0,
                    value,
                    replacement.fill,
                )?;
                let destination = self.span(
                    replacement.replacement_bundle,
                    layer,
                    0,
                    value,
                    replacement.fill,
                )?;
                self.storage.copy_within(source, destination.start);
            }
        }
        Ok(())
    }
    fn span(
        &self,
        bundle: usize,
        layer: usize,
        token: usize,
        value: bool,
        rows: usize,
    ) -> Result<Range<usize>> {
        let page_tokens = self.ledger.plan.page_tokens.count();
        if bundle >= self.ledger.plan.allocation.bundle_count
            || layer >= self.ledger.plan.geometry.layers
            || token >= page_tokens
            || rows > page_tokens - token
        {
            return PagedLayoutSnafu {
                operation: "page addressing",
            }
            .fail();
        }
        let bundle_start = bundle
            .checked_mul(self.ledger.plan.bundle_elements()?)
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "bundle offset",
                }
                .build()
            })?;
        let layer_stride = page_tokens
            .checked_mul(self.ledger.plan.geometry.row_width)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "layer stride",
                }
                .build()
            })?;
        let value_offset = if value {
            page_tokens
                .checked_mul(self.ledger.plan.geometry.row_width)
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
                    .checked_mul(self.ledger.plan.geometry.row_width)
                    .and_then(|y| x.checked_add(y))
            })
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "row offset",
                }
                .build()
            })?;
        let end = rows
            .checked_mul(self.ledger.plan.geometry.row_width)
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
    reservation: AppendReservation,
    committed: bool,
}

/// Verified CPU append publication that still rolls back when dropped.
#[derive(Debug)]
pub struct PagedPreparedCommit<'a> {
    append: PagedAppend<'a>,
}

#[derive(Debug)]
struct PagedWriteLocation {
    page: usize,
    within: usize,
    key: Range<usize>,
    value: Range<usize>,
}

#[derive(Clone, Copy, Debug)]
struct LedgerWriteLocation {
    page: usize,
    bundle: usize,
    within: usize,
}

/// Unpublished logical mutations for one append transaction.
#[derive(Debug)]
struct AppendReservation {
    append_tokens: usize,
    original_tokens: usize,
    target_tokens: usize,
    original_page_count: usize,
    replaced_tail: Option<TailReplacement>,
}

/// Cache-owned native publication state after all layers have been verified.
///
/// This contains no device handles or independent ledger: it parks the one
/// reservation that already mutated the owning pool's host ledger.
#[cfg(any(feature = "gpu", test))]
#[derive(Debug, Default)]
struct NativePreparedCommit {
    reservation: Option<AppendReservation>,
}

#[cfg(any(feature = "gpu", test))]
impl NativePreparedCommit {
    fn ensure_empty(&self) -> Result<()> {
        if self.reservation.is_some() {
            return PagedNativeCommitPreparedSnafu.fail();
        }
        Ok(())
    }

    fn park_verified(&mut self, reservation: AppendReservation) {
        self.reservation = Some(reservation);
    }

    fn take(&mut self) -> Result<AppendReservation> {
        self.reservation
            .take()
            .ok_or_else(|| PagedNativeCommitNotPreparedSnafu.build())
    }
}

impl<'a> PagedAppend<'a> {
    /// Write one contiguous transaction-relative K/V row for one layer.
    pub fn write_layer_row(
        &mut self,
        layer: usize,
        token: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        let location = self.write_location(layer, token, keys, values)?;
        self.copy_row(location.key, keys, "key row write")?;
        self.copy_row(location.value, values, "value row write")?;
        self.record_write(layer, location.page, location.within)
    }
    fn write_location(
        &self,
        layer: usize,
        token: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<PagedWriteLocation> {
        self.check_row(layer, "key", keys)?;
        self.check_row(layer, "value", values)?;
        let location = self
            .pool
            .ledger
            .write_location(&self.reservation, layer, token)?;
        let key = self
            .pool
            .span(location.bundle, layer, location.within, false, 1)?;
        let value = self
            .pool
            .span(location.bundle, layer, location.within, true, 1)?;
        self.check_storage_row(&key, "key row write")?;
        self.check_storage_row(&value, "value row write")?;
        Ok(PagedWriteLocation {
            page: location.page,
            within: location.within,
            key,
            value,
        })
    }
    fn check_storage_row(&self, row: &Range<usize>, operation: &'static str) -> Result<()> {
        self.pool
            .storage
            .get(row.clone())
            .map(|_| ())
            .ok_or_else(|| PagedLayoutSnafu { operation }.build())
    }
    fn copy_row(
        &mut self,
        row: Range<usize>,
        values: &[f32],
        operation: &'static str,
    ) -> Result<()> {
        self.pool
            .storage
            .get_mut(row)
            .ok_or_else(|| PagedLayoutSnafu { operation }.build())?
            .copy_from_slice(values);
        Ok(())
    }
    fn record_write(&mut self, layer: usize, page: usize, within: usize) -> Result<()> {
        self.pool.ledger.record_write(layer, page, within)
    }
    /// View only committed rows plus this layer's contiguous written prefix.
    pub fn layer_kv(&self, layer: usize) -> Result<PagedLayerKv<'_>> {
        let tokens = self.pool.ledger.visible_tokens(&self.reservation, layer)?;
        Ok(PagedLayerKv {
            pool: self.pool,
            layer,
            tokens,
        })
    }
    /// Verify every layer before retaining this append for infallible publication.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::PagedIncompleteAppend`] when any layer lacks an active
    /// transaction row. On error, consuming this append restores its staged
    /// logical state through the existing rollback guard.
    pub fn prepare_commit(self) -> Result<PagedPreparedCommit<'a>> {
        self.pool.ledger.validate_commit(&self.reservation)?;
        Ok(PagedPreparedCommit { append: self })
    }

    /// Validate and publish this append through the prepared-commit path.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::PagedIncompleteAppend`] when any layer lacks an active
    /// transaction row. On error, the append rollback guard restores staged
    /// logical state.
    pub fn commit(self) -> Result<()> {
        self.prepare_commit()?.commit();
        Ok(())
    }
    fn check_row(&self, layer: usize, kind: &'static str, row: &[f32]) -> Result<()> {
        if row.len() != self.pool.ledger.plan.geometry.row_width {
            return PagedRowWidthSnafu {
                kind,
                layer,
                actual: row.len(),
                expected: self.pool.ledger.plan.geometry.row_width,
            }
            .fail();
        }
        Ok(())
    }
}

impl PagedPreparedCommit<'_> {
    /// Publish this previously verified append without another fallible step.
    pub fn commit(mut self) {
        self.append
            .pool
            .ledger
            .publish_commit(&mut self.append.reservation);
        self.append.committed = true;
    }
}

impl Drop for PagedAppend<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        self.pool.ledger.rollback(&self.reservation);
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
        let page_tokens = self.pool.ledger.plan.page_tokens.count();
        let page = token / page_tokens;
        let within = token % page_tokens;
        let fill = *self.pool.ledger.fills.get(page).ok_or_else(|| {
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
        let bundle = *self.pool.ledger.table.get(page).ok_or_else(|| {
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

/// Explicit native physical backing derived from one logical paged geometry.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy)]
pub struct NativePagedKvPlan {
    logical: PagedKvPlan,
    kv_heads: usize,
    head_width: usize,
    layout: kernels::paged_kv::PagedKvNativeLayout,
}

#[cfg(feature = "gpu")]
impl NativePagedKvPlan {
    /// Bind native K/V rows to the shared logical geometry with an explicit page size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PagedArithmetic`] or [`Error::PagedLayout`] when the
    /// native row axes cannot exactly bind the logical row width, or when the
    /// shared logical allocation cannot be derived. Returns
    /// [`Error::Kernel`] when the checked native backing descriptor rejects
    /// its dimensions or layout.
    pub fn try_from_geometry(
        geometry: PagedKvGeometry,
        kv_heads: usize,
        head_width: usize,
        page_tokens: NativePageTokens,
    ) -> Result<Self> {
        let row_width = kv_heads.checked_mul(head_width).ok_or_else(|| {
            PagedArithmeticSnafu {
                operation: "native key-value row width",
            }
            .build()
        })?;
        if row_width != geometry.row_width {
            return PagedLayoutSnafu {
                operation: "native key-value heads and width",
            }
            .fail();
        }
        let logical = PagedKvPlan::new(geometry, page_tokens.into())?;
        let layout = kernels::paged_kv::PagedKvNativeLayout::try_from_dimensions(
            geometry.layers,
            geometry.row_width,
            page_tokens.get(),
            logical.allocation.bundle_count,
        )?;
        Ok(Self {
            logical,
            kv_heads,
            head_width,
            layout,
        })
    }

    /// Shared logical plan used by CPU-style page ownership and transactions.
    #[must_use]
    pub const fn logical(self) -> PagedKvPlan {
        self.logical
    }

    /// Explicit native K/V backing geometry.
    #[must_use]
    pub const fn layout(self) -> kernels::paged_kv::PagedKvNativeLayout {
        self.layout
    }

    /// Key/value heads represented by one native row.
    #[must_use]
    pub const fn kv_heads(self) -> usize {
        self.kv_heads
    }

    /// Scalars in each key/value head.
    #[must_use]
    pub const fn head_width(self) -> usize {
        self.head_width
    }

    /// Return the exact f32 elements in each key or value backing.
    #[must_use]
    pub const fn key_value_elements(self) -> usize {
        self.layout.backing_elements()
    }

    /// Return the exact u32 elements in the native page table.
    #[must_use]
    pub const fn page_table_elements(self) -> usize {
        self.logical.allocation.page_count
    }
}

#[cfg(feature = "gpu")]
impl From<NativePageTokens> for PageTokens {
    fn from(value: NativePageTokens) -> Self {
        match value {
            NativePageTokens::B8 => Self::B8,
            NativePageTokens::B16 => Self::B16,
            NativePageTokens::B32 => Self::B32,
        }
    }
}

/// Optional native K/V backing for one non-cloneable session.
///
/// The host ledger remains authoritative. Device K/V and table allocations are
/// physical mirrors only; a caller that observes a failure after submission
/// must poison and retain the entire owning session until completion is known.
#[cfg(feature = "gpu")]
pub struct NativePagedKvPool {
    ledger: PagedKvLedger,
    prepared: NativePreparedCommit,
    plan: NativePagedKvPlan,
    keys: DeviceBuffer<f32>,
    values: DeviceBuffer<f32>,
    table: DeviceBuffer<u32>,
    poisoned: bool,
}

/// Caller-owned native K/V buffers awaiting one checked pool binding.
///
/// This is an ownership carrier only. It performs no allocation, upload, or
/// device work; [`NativePagedKvPool::try_from_buffers`] validates its exact
/// geometry and device identity while returning these same owners on refusal.
#[cfg(feature = "gpu")]
pub struct NativePagedKvBuffers {
    keys: DeviceBuffer<f32>,
    values: DeviceBuffer<f32>,
    table: DeviceBuffer<u32>,
}

#[cfg(feature = "gpu")]
impl NativePagedKvBuffers {
    /// Assemble caller-owned native K/V and page-table buffers without validation.
    #[must_use]
    pub fn new(
        keys: DeviceBuffer<f32>,
        values: DeviceBuffer<f32>,
        table: DeviceBuffer<u32>,
    ) -> Self {
        Self {
            keys,
            values,
            table,
        }
    }

    /// Consume this carrier and return its original typed buffer owners.
    #[must_use]
    pub fn into_parts(self) -> (DeviceBuffer<f32>, DeviceBuffer<f32>, DeviceBuffer<u32>) {
        (self.keys, self.values, self.table)
    }
}

/// A rejected native K/V buffer binding that still owns every caller buffer.
#[cfg(feature = "gpu")]
pub struct NativePagedKvPoolBindingError {
    error: Box<crate::Error>,
    buffers: NativePagedKvBuffers,
}

#[cfg(feature = "gpu")]
impl NativePagedKvPoolBindingError {
    /// Borrow the checked reason the binding was refused.
    #[must_use]
    pub fn error(&self) -> &crate::Error {
        &self.error
    }

    /// Consume this rejection and recover the error and original buffer owners.
    #[must_use]
    pub fn into_parts(self) -> (crate::Error, NativePagedKvBuffers) {
        (*self.error, self.buffers)
    }
}

#[cfg(feature = "gpu")]
impl NativePagedKvPool {
    /// Allocate native K/V and table mirrors before any append is admitted.
    ///
    /// # Errors
    ///
    /// Returns typed HIP allocation errors, or the shared ledger's checked
    /// geometry and metadata-allocation errors, without publishing an append.
    pub fn new(plan: NativePagedKvPlan, device: &Device) -> Result<Self> {
        let ledger = PagedKvLedger::new(plan.logical)?;
        let keys = DeviceBuffer::alloc(device, plan.key_value_elements())?;
        let values = DeviceBuffer::alloc(device, plan.key_value_elements())?;
        let table = DeviceBuffer::alloc(device, plan.page_table_elements())?;
        Ok(Self {
            ledger,
            prepared: NativePreparedCommit::default(),
            plan,
            keys,
            values,
            table,
            poisoned: false,
        })
    }

    /// Bind caller-owned exact native buffers to this checked pool plan.
    ///
    /// On refusal this consumes no buffer: the returned
    /// [`NativePagedKvPoolBindingError`] retains the same typed owners for an
    /// explicit construction-failure teardown path.
    pub fn try_from_buffers(
        plan: NativePagedKvPlan,
        buffers: NativePagedKvBuffers,
    ) -> core::result::Result<Self, NativePagedKvPoolBindingError> {
        let binding = (|| -> Result<PagedKvLedger> {
            if buffers.keys.len() != plan.key_value_elements()
                || buffers.values.len() != plan.key_value_elements()
                || buffers.table.len() != plan.page_table_elements()
            {
                return PagedLayoutSnafu {
                    operation: "native paged-K/V caller buffer length",
                }
                .fail();
            }
            ensure_same_process_device(
                buffers.keys.device().ordinal(),
                buffers.values.device().ordinal(),
            )?;
            ensure_same_process_device(
                buffers.keys.device().ordinal(),
                buffers.table.device().ordinal(),
            )?;
            PagedKvLedger::new(plan.logical)
        })();
        let ledger = match binding {
            Ok(ledger) => ledger,
            Err(error) => {
                return Err(NativePagedKvPoolBindingError {
                    error: Box::new(error),
                    buffers,
                });
            }
        };
        let NativePagedKvBuffers {
            keys,
            values,
            table,
        } = buffers;
        Ok(Self {
            ledger,
            prepared: NativePreparedCommit::default(),
            plan,
            keys,
            values,
            table,
            poisoned: false,
        })
    }

    /// Consume this pool and return its original typed device buffers.
    ///
    /// This performs no synchronization, HIP release, or eviction
    /// acknowledgement. A caller handling submitted work must retain the
    /// returned owners in a quiescing teardown path before they can drop.
    #[must_use]
    pub fn into_buffers(self) -> NativePagedKvBuffers {
        NativePagedKvBuffers {
            keys: self.keys,
            values: self.values,
            table: self.table,
        }
    }

    /// Stage a native append and mirror its required COW/table mutations.
    ///
    /// # Safety
    ///
    /// The caller owns the stream and every device allocation participating in
    /// the complete layer. `stream` must belong to this pool's device and be
    /// the ordered stream used for every prepare, row, and attention operation
    /// in this transaction. On any submission or completion uncertainty it
    /// must retain that complete bundle and never reuse this pool.
    ///
    /// # Errors
    ///
    /// Returns a typed poisoned/prepared-state, device-mismatch, append
    /// capacity, ledger, or native copy/table-kernel failure. Host staging is
    /// unpublished on failure; submitted native backing remains poisoned.
    pub unsafe fn begin_append(
        &mut self,
        append_tokens: usize,
        stream: &Stream,
    ) -> Result<NativePagedAppend<'_>> {
        self.ensure_not_poisoned()?;
        self.prepared.ensure_empty()?;
        self.ensure_stream_device(stream)?;
        let reservation = self.ledger.begin_append(append_tokens)?;
        let prepared = unsafe { self.prepare_device_append(&reservation, stream) };
        if let Err(error) = prepared {
            // Host state is still authoritative and unpublished.  Device
            // writes may be uncertain, so restore only the host ledger and
            // permanently refuse reuse of this native backing.
            self.ledger.rollback(&reservation);
            self.poisoned = true;
            return Err(error);
        }
        Ok(NativePagedAppend {
            pool: self,
            reservation: Some(reservation),
        })
    }

    /// Publish the one prepared native append after external completion proof.
    ///
    /// # Safety
    ///
    /// The caller must have successfully synchronized the ordered stream used
    /// for that append's prepare, row, and attention submissions. No read or
    /// write of its K/V, table, query, or output buffers may remain pending.
    /// On any submission or completion failure, retain and never reuse this
    /// complete native session instead of calling this method.
    ///
    /// # Errors
    ///
    /// Returns a typed poisoned or missing-prepared-append error before any
    /// host-ledger publication.
    pub unsafe fn commit_prepared_after_completion(&mut self) -> Result<()> {
        self.ensure_not_poisoned()?;
        let mut reservation = self.prepared.take()?;
        self.ledger.publish_commit(&mut reservation);
        Ok(())
    }

    fn ensure_not_poisoned(&self) -> Result<()> {
        if self.poisoned {
            return PagedNativePoisonedSnafu.fail();
        }
        Ok(())
    }

    fn ensure_stream_device(&self, stream: &Stream) -> Result<()> {
        ensure_same_process_device(self.keys.device().ordinal(), stream.device().ordinal())
    }

    fn ensure_buffer_device(&self, buffer: &DeviceBuffer<f32>) -> Result<()> {
        ensure_same_process_device(self.keys.device().ordinal(), buffer.device().ordinal())
    }

    unsafe fn prepare_device_append(
        &mut self,
        reservation: &AppendReservation,
        stream: &Stream,
    ) -> Result<()> {
        if let Some(replacement) = reservation.replaced_tail {
            // SAFETY: pool-owned non-overlapping K/V spans and reservation-derived pages remain live through caller-owned completion.
            unsafe {
                kernels::paged_kv::copy_tail_f32(
                    self.plan.layout,
                    self.keys.as_device_ptr(),
                    self.keys.len(),
                    self.values.as_device_ptr(),
                    self.values.len(),
                    replacement.original_bundle,
                    replacement.replacement_bundle,
                    replacement.fill,
                    stream,
                )
            }?;
            // SAFETY: the ledger selected this in-range logical/physical page mapping.
            unsafe {
                kernels::paged_kv::write_table_u32(
                    self.table.as_device_ptr(),
                    self.table.len(),
                    replacement.page,
                    replacement.replacement_bundle,
                    stream,
                )
            }?;
        }
        for logical_page in reservation.original_page_count..self.ledger.table.len() {
            let physical_page = *self.ledger.table.get(logical_page).ok_or_else(|| {
                PagedLayoutSnafu {
                    operation: "native staged page table",
                }
                .build()
            })?;
            // SAFETY: the ledger's checked table range supplies both ABI-sized indices.
            unsafe {
                kernels::paged_kv::write_table_u32(
                    self.table.as_device_ptr(),
                    self.table.len(),
                    logical_page,
                    physical_page,
                    stream,
                )
            }?;
        }
        Ok(())
    }
}

/// Borrowed native append transaction with unpublished host-ledger mutations.
#[cfg(feature = "gpu")]
pub struct NativePagedAppend<'a> {
    pool: &'a mut NativePagedKvPool,
    reservation: Option<AppendReservation>,
}

#[cfg(feature = "gpu")]
impl NativePagedAppend<'_> {
    /// Stage every active `[tokens, kv_heads, head_width]` row for one layer.
    ///
    /// # Errors
    ///
    /// Returns a typed cache, device, capacity-prefix, row-geometry, or HIP
    /// append-submission failure. Submission failures poison the all-layer
    /// prepared transaction.
    ///
    /// # Safety
    ///
    /// The capacity-sized input owners and stream must remain live on this
    /// pool's device through completion. A submitted failure poisons the pool.
    pub unsafe fn write_layer_rows(
        &mut self,
        layer: usize,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        self.pool.ensure_not_poisoned()?;
        self.pool.ensure_stream_device(stream)?;
        self.pool.ensure_buffer_device(keys)?;
        self.pool.ensure_buffer_device(values)?;
        let append_tokens = self.reservation()?.append_tokens;
        let active = append_tokens
            .checked_mul(self.pool.plan.logical.geometry.row_width)
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "native append active row prefix",
                }
                .build()
            })?;
        if keys.len() < active || values.len() < active {
            return PagedLayoutSnafu {
                operation: "native append active row prefix",
            }
            .fail();
        }
        for token in 0..append_tokens {
            let location = {
                let reservation = self.reservation()?;
                self.pool.ledger.write_location(reservation, layer, token)?
            };
            let input_offset = token
                .checked_mul(self.pool.plan.logical.geometry.row_width)
                .ok_or_else(|| {
                    PagedArithmeticSnafu {
                        operation: "native append input row offset",
                    }
                    .build()
                })?;
            // SAFETY: active-prefix validation and checked offset select one row in each caller owner.
            let submitted = unsafe {
                kernels::paged_kv::append_row_f32(
                    self.pool.plan.layout,
                    self.pool.keys.as_device_ptr(),
                    self.pool.keys.len(),
                    self.pool.values.as_device_ptr(),
                    self.pool.values.len(),
                    keys.as_device_ptr().add(input_offset),
                    self.pool.plan.logical.geometry.row_width,
                    values.as_device_ptr().add(input_offset),
                    self.pool.plan.logical.geometry.row_width,
                    layer,
                    location.bundle,
                    location.within,
                    stream,
                )
            };
            if let Err(error) = submitted {
                self.pool.poisoned = true;
                return Err(error.into());
            }
            if let Err(error) = self
                .pool
                .ledger
                .record_write(layer, location.page, location.within)
            {
                self.pool.poisoned = true;
                return Err(error);
            }
        }
        Ok(())
    }

    /// Stage one device-resident K/V row for one full-attention layer.
    ///
    /// # Safety
    ///
    /// `keys` and `values` must remain live and immutable through stream
    /// completion. Their device must match
    /// this pool and `stream`, which must be the ordered pool-device stream passed to
    /// [`NativePagedKvPool::begin_append`]. The caller owns completion and
    /// must poison its whole session if submission completion becomes uncertain.
    ///
    /// # Errors
    ///
    /// Returns a typed poisoned-state, device-mismatch, row-shape/order, or
    /// native append-kernel error. A failed submitted append permanently
    /// poisons the native backing.
    pub unsafe fn write_layer_row(
        &mut self,
        layer: usize,
        token: usize,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        self.pool.ensure_not_poisoned()?;
        self.pool.ensure_stream_device(stream)?;
        self.pool.ensure_buffer_device(keys)?;
        self.pool.ensure_buffer_device(values)?;
        let reservation = self.reservation()?;
        let location = self.pool.ledger.write_location(reservation, layer, token)?;
        let submitted = unsafe {
            kernels::paged_kv::append_row_f32(
                self.pool.plan.layout,
                self.pool.keys.as_device_ptr(),
                self.pool.keys.len(),
                self.pool.values.as_device_ptr(),
                self.pool.values.len(),
                keys.as_device_ptr(),
                keys.len(),
                values.as_device_ptr(),
                values.len(),
                layer,
                location.bundle,
                location.within,
                stream,
            )
        };
        if let Err(error) = submitted {
            self.pool.poisoned = true;
            return Err(error.into());
        }
        let recorded = self
            .pool
            .ledger
            .record_write(layer, location.page, location.within);
        if recorded.is_err() {
            // WHY: a row append was already submitted, so a logical error
            // cannot make the native backing safely reusable.
            self.pool.poisoned = true;
        }
        recorded
    }

    /// Return one staged native layer view for Q=1 paged attention.
    ///
    /// # Errors
    ///
    /// Returns a typed missing-reservation, layer-range, or visible-prefix
    /// failure without exposing native backing pointers.
    pub fn layer_kv(&self, layer: usize) -> Result<NativePagedLayerKv<'_>> {
        let reservation = self.reservation()?;
        let tokens = self.pool.ledger.visible_tokens(reservation, layer)?;
        Ok(NativePagedLayerKv {
            pool: self.pool,
            layer,
            tokens,
            offset: reservation.original_tokens,
        })
    }

    /// Verify every layer's staged rows and park this append in its pool.
    ///
    /// This ends the borrow so an external resource owner can synchronize the
    /// complete device bundle before calling
    /// [`NativePagedKvPool::commit_prepared_after_completion`]. It does not
    /// publish host ledger state or issue a device operation.
    ///
    /// # Errors
    ///
    /// Returns a typed poisoned/prepared-state or incomplete-all-layer append
    /// error, leaving host publication unchanged.
    pub fn prepare_commit(mut self) -> Result<()> {
        self.pool.ensure_not_poisoned()?;
        self.pool.prepared.ensure_empty()?;
        self.pool.ledger.validate_commit(self.reservation()?)?;
        let reservation = self.reservation.take().ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "native append reservation",
            }
            .build()
        })?;
        self.pool.prepared.park_verified(reservation);
        Ok(())
    }

    fn reservation(&self) -> Result<&AppendReservation> {
        self.reservation.as_ref().ok_or_else(|| {
            PagedLayoutSnafu {
                operation: "native append reservation",
            }
            .build()
        })
    }
}

#[cfg(feature = "gpu")]
impl Drop for NativePagedAppend<'_> {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.as_ref() {
            // WHY: logical staging is unpublished; no device rollback is safe after any submission.
            self.pool.ledger.rollback(reservation);
            // A prepare/table or row kernel may already be in flight.  Keep
            // the host ledger unpublished, but refuse any reuse until the
            // caller has retained and resolved the whole native session.
            self.pool.poisoned = true;
        }
    }
}

/// Opaque per-layer native K/V source for the Q=1 attention launcher.
#[cfg(feature = "gpu")]
pub struct NativePagedLayerKv<'a> {
    pool: &'a NativePagedKvPool,
    layer: usize,
    tokens: usize,
    offset: usize,
}

#[cfg(feature = "gpu")]
struct NativePagedDecodeBacking {
    keys: *const f32,
    values: *const f32,
    elements: usize,
    table: *const u32,
    table_entries: usize,
}

#[cfg(feature = "gpu")]
impl NativePagedLayerKv<'_> {
    /// Rows visible to this layer during its current append.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Launch Q=1 paged attention without exposing cache table or backing pointers.
    ///
    /// # Safety
    ///
    /// `query` must remain live and immutable through completion. `output`
    /// must remain live and exclusively owned through completion; both buffers
    /// must be on this pool's device. The query, staged K/V and every arithmetic
    /// intermediate must satisfy the attention launcher's finite normal-or-zero
    /// numerical domain. Buffer ownership and device checks do not prove that
    /// numerical precondition.
    /// `stream` must be this pool's device stream and exactly the ordered
    /// stream passed to begin/row submission for this transaction.
    ///
    /// # Errors
    ///
    /// Returns a typed device-mismatch, descriptor-binding, checked-offset,
    /// or native attention-launch failure before or during submission.
    pub unsafe fn launch_paged_decode(
        &self,
        plan: NativePagedDecodePlan,
        query: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        let backing = self.checked_attention_backing(plan, query, output, stream)?;
        // SAFETY: layer-major layout gives exactly one dense physical-page K/V span; all raw launch obligations remain caller-owned.
        unsafe {
            kernels::attention::launch_paged_decode_q1_f32(
                plan,
                query.as_device_ptr(),
                query.len(),
                backing.keys,
                backing.elements,
                backing.values,
                backing.elements,
                backing.table,
                backing.table_entries,
                output.as_device_ptr(),
                output.len(),
                stream,
            )
        }?;
        Ok(())
    }

    /// Launch checked Q=1 paged attention without exposing cache pointers.
    ///
    /// # Safety
    ///
    /// `query` must remain live and immutable through completion. `output`
    /// must remain live and exclusively owned through completion. Both buffers
    /// and `status` must belong to this pool's device. `stream` must be this
    /// pool's ordered transaction stream. The status records explicit operand
    /// and arithmetic faults, but does not establish table-value validity,
    /// device math-mode qualification, or hardware numerical parity.
    ///
    /// # Errors
    ///
    /// Returns a typed device-mismatch, descriptor-binding, checked-offset, or
    /// checked native attention-launch failure. The caller must synchronize and
    /// read `status` before publishing the append's host ledger.
    pub unsafe fn launch_paged_decode_checked(
        &self,
        plan: NativePagedDecodePlan,
        query: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: &Stream,
        status: &NativeNumericalStatus,
    ) -> Result<()> {
        let backing = self.checked_attention_backing(plan, query, output, stream)?;
        // SAFETY: the opaque backing helper established exact spans; the caller
        // retains every buffer and status allocation through completion.
        unsafe {
            kernels::attention::launch_paged_decode_q1_f32_checked(
                plan,
                query.as_device_ptr(),
                query.len(),
                backing.keys,
                backing.elements,
                backing.values,
                backing.elements,
                backing.table,
                backing.table_entries,
                output.as_device_ptr(),
                output.len(),
                stream,
                status,
            )
        }?;
        Ok(())
    }

    /// Launch checked B=1 causal prefill without exposing cache pointers.
    ///
    /// # Errors
    ///
    /// Returns a typed cache, device, final-visible-prefix, page-geometry, or
    /// checked native attention-launch failure. The caller must synchronize and
    /// read `status` before publishing the append's host ledger.
    ///
    /// # Safety
    ///
    /// Query, output, status, and this staged cache must remain live on the
    /// ordered pool stream through completion. The caller reads status before
    /// publication.
    pub unsafe fn launch_paged_prefill_checked(
        &self,
        plan: NativePagedPrefillPlan,
        query: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: &Stream,
        status: &NativeNumericalStatus,
    ) -> Result<()> {
        let backing = self.checked_prefill_backing(plan, query, output, stream)?;
        // SAFETY: opaque backing and typed owners remain live through caller completion.
        unsafe {
            kernels::attention::launch_paged_prefill_b1_f32_checked(
                plan,
                query.as_device_ptr(),
                query.len(),
                backing.keys,
                backing.elements,
                backing.values,
                backing.elements,
                backing.table,
                backing.table_entries,
                output.as_device_ptr(),
                output.len(),
                stream,
                status,
            )
        }?;
        Ok(())
    }

    fn checked_prefill_backing(
        &self,
        plan: NativePagedPrefillPlan,
        query: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<NativePagedDecodeBacking> {
        self.pool.ensure_stream_device(stream)?;
        self.pool.ensure_buffer_device(query)?;
        self.pool.ensure_buffer_device(output)?;
        validate_native_prefill_binding(self.pool.plan, self.offset, self.tokens, plan)?;
        let layer_offset = self
            .layer
            .checked_mul(self.pool.plan.layout.layer_elements())
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "native prefill layer backing offset",
                }
                .build()
            })?;
        // SAFETY: checked layer index identifies one full layer backing extent.
        let keys = unsafe { self.pool.keys.as_device_ptr().add(layer_offset) }.cast_const();
        // SAFETY: K/V layer extents are equal and non-overlapping owners.
        let values = unsafe { self.pool.values.as_device_ptr().add(layer_offset) }.cast_const();
        Ok(NativePagedDecodeBacking {
            keys,
            values,
            elements: self.pool.plan.layout.layer_elements(),
            table: self.pool.table.as_device_ptr().cast_const(),
            table_entries: plan.page_table_entries(),
        })
    }

    fn checked_attention_backing(
        &self,
        plan: NativePagedDecodePlan,
        query: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<NativePagedDecodeBacking> {
        self.pool.ensure_stream_device(stream)?;
        self.pool.ensure_buffer_device(query)?;
        self.pool.ensure_buffer_device(output)?;
        validate_native_attention_binding(self.pool.plan, self.tokens, plan)?;
        let layer_offset = self
            .layer
            .checked_mul(self.pool.plan.layout.layer_elements())
            .ok_or_else(|| {
                PagedArithmeticSnafu {
                    operation: "native layer backing offset",
                }
                .build()
            })?;
        // SAFETY: `layer_kv` admits a ledger-valid layer, and the checked
        // offset identifies one whole layer extent in each owned backing.
        let keys = unsafe { self.pool.keys.as_device_ptr().add(layer_offset) }.cast_const();
        // SAFETY: keys and values have the same checked layer-major extent.
        let values = unsafe { self.pool.values.as_device_ptr().add(layer_offset) }.cast_const();
        Ok(NativePagedDecodeBacking {
            keys,
            values,
            elements: self.pool.plan.layout.layer_elements(),
            table: self.pool.table.as_device_ptr().cast_const(),
            table_entries: plan.page_table_entries(),
        })
    }
}

#[cfg(feature = "gpu")]
fn ensure_same_process_device(expected: std::ffi::c_int, actual: std::ffi::c_int) -> Result<()> {
    if expected != actual {
        return PagedNativeDeviceMismatchSnafu { expected, actual }.fail();
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn validate_native_attention_binding(
    cache: NativePagedKvPlan,
    visible_tokens: usize,
    attention: NativePagedDecodePlan,
) -> Result<()> {
    let logical = attention.logical();
    if visible_tokens != logical.visible_tokens()
        || cache.layout.physical_pages() != attention.physical_pages()
        || cache.layout.page_tokens() != attention.page_tokens().get()
        || cache.kv_heads != logical.kv_heads()
        || cache.head_width != logical.head_width()
    {
        return PagedLayoutSnafu {
            operation: "native paged-attention descriptor binding",
        }
        .fail();
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn validate_native_prefill_binding(
    cache: NativePagedKvPlan,
    append_offset: usize,
    visible_tokens: usize,
    attention: NativePagedPrefillPlan,
) -> Result<()> {
    let logical = attention.logical();
    if append_offset != logical.offset()
        || visible_tokens != logical.visible_tokens()
        || cache.layout.physical_pages() != attention.physical_pages()
        || cache.layout.page_tokens() != attention.page_tokens().get()
        || cache.kv_heads != logical.kv_heads()
        || cache.head_width != logical.head_width()
    {
        return PagedLayoutSnafu {
            operation: "native paged-prefill descriptor binding",
        }
        .fail();
    }
    Ok(())
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
        for (layer, rows) in model.iter_mut().enumerate() {
            for relative in 0..tokens {
                rows.push(model_row(layer, start + relative));
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

    fn record_all_ledger_rows(
        ledger: &mut PagedKvLedger,
        reservation: &AppendReservation,
    ) -> Result<()> {
        for layer in 0..LAYERS {
            for token in 0..reservation.append_tokens {
                let location = ledger.write_location(reservation, layer, token)?;
                ledger.record_write(layer, location.page, location.within)?;
            }
        }
        Ok(())
    }

    fn assert_inventory(pool: &PagedKvPool, held_old_tail: Option<usize>) {
        assert_eq!(pool.ledger.table.len(), pool.ledger.fills.len());
        let mut seen = BTreeSet::new();
        for bundle in &pool.ledger.table {
            assert!(*bundle < pool.ledger.plan.allocation.bundle_count);
            assert!(seen.insert(*bundle));
        }
        for bundle in &pool.ledger.free {
            assert!(*bundle < pool.ledger.plan.allocation.bundle_count);
            assert!(seen.insert(*bundle));
        }
        if let Some(bundle) = held_old_tail {
            assert!(bundle < pool.ledger.plan.allocation.bundle_count);
            assert!(seen.insert(bundle));
        }
        assert_eq!(seen.len(), pool.ledger.plan.allocation.bundle_count);
        assert_eq!(
            pool.ledger.table.len() + pool.ledger.free.len() + usize::from(held_old_tail.is_some()),
            pool.ledger.plan.allocation.bundle_count
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
        assert_eq!(pool.ledger.committed_tokens, tokens);
        for (layer, expected_rows) in model.iter().enumerate() {
            let rows = pool.layer_kv(layer)?;
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
            txn.reservation
                .replaced_tail
                .map(|replacement| replacement.original_bundle),
        );
    }

    struct TailWitness {
        bundle: usize,
        fill: usize,
        payload: Vec<Vec<f32>>,
    }

    fn seed_partial_tail(
        pool: &mut PagedKvPool,
        model: &mut [Vec<ModelRow>; LAYERS],
        start: usize,
    ) -> Result<()> {
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
        append_model(model, 0, start);
        assert_inventory(pool, None);
        assert_committed_matches(pool, model)?;
        Ok(())
    }

    fn tail_witness(pool: &PagedKvPool) -> Result<TailWitness> {
        let bundle = pool.ledger.table[0];
        let fill = pool.ledger.fills[0];
        Ok(TailWitness {
            bundle,
            fill,
            payload: tail_payload(pool, bundle, fill)?,
        })
    }

    fn exercise_incomplete_partial_append(
        pool: &mut PagedKvPool,
        start: usize,
        witness: &TailWitness,
    ) -> Result<()> {
        {
            let mut incomplete = pool.begin_append(2)?;
            assert!(incomplete.reservation.replaced_tail.is_some());
            assert_ne!(incomplete.pool.ledger.table[0], witness.bundle);
            assert_staged_inventory(&incomplete);
            assert_tail_payload(
                incomplete.pool,
                witness.bundle,
                witness.fill,
                &witness.payload,
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
                witness.bundle,
                witness.fill,
                &witness.payload,
            )?;
            assert_staged_inventory(&incomplete);
            assert!(matches!(
                incomplete.prepare_commit(),
                Err(Error::PagedIncompleteAppend { .. })
            ));
        }
        Ok(())
    }

    fn assert_tail_restored(
        pool: &PagedKvPool,
        model: &[Vec<ModelRow>; LAYERS],
        witness: &TailWitness,
    ) -> Result<()> {
        assert_eq!(pool.ledger.table[0], witness.bundle);
        assert_tail_payload(pool, witness.bundle, witness.fill, &witness.payload)?;
        assert_inventory(pool, None);
        assert_committed_matches(pool, model)
    }

    fn drop_partial_append(pool: &mut PagedKvPool, start: usize) -> Result<()> {
        {
            let mut dropped = pool.begin_append(2)?;
            write_layer(&mut dropped, 0, start, 2)?;
            assert_staged_inventory(&dropped);
        }
        Ok(())
    }

    fn retry_partial_append(pool: &mut PagedKvPool, start: usize) -> Result<()> {
        {
            let mut retry = pool.begin_append(2)?;
            write_all(&mut retry, start, 2)?;
            assert_staged_inventory(&retry);
            retry.commit()?;
        }
        Ok(())
    }

    fn prepare_all<'a>(
        pool: &'a mut PagedKvPool,
        start: usize,
        tokens: usize,
    ) -> Result<PagedPreparedCommit<'a>> {
        let mut append = pool.begin_append(tokens)?;
        write_all(&mut append, start, tokens)?;
        append.prepare_commit()
    }

    fn exercise_partial_tail_rollback(page_tokens: usize, plan: PagedKvPlan) -> Result<()> {
        let mut pool = PagedKvPool::new(plan)?;
        let mut model = empty_model();
        let start = page_tokens - 1;
        assert_inventory(&pool, None);
        seed_partial_tail(&mut pool, &mut model, start)?;
        let witness = tail_witness(&pool)?;
        exercise_incomplete_partial_append(&mut pool, start, &witness)?;
        assert_tail_restored(&pool, &model, &witness)?;
        drop_partial_append(&mut pool, start)?;
        assert_tail_restored(&pool, &model, &witness)?;
        retry_partial_append(&mut pool, start)?;
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

        let old_tail = if start.is_multiple_of(page_tokens) {
            None
        } else {
            let page = pool.ledger.table.len() - 1;
            let bundle = pool.ledger.table[page];
            Some((
                bundle,
                pool.ledger.fills[page],
                tail_payload(&pool, bundle, pool.ledger.fills[page])?,
            ))
        };
        {
            let mut txn = pool.begin_append(append_tokens)?;
            assert_staged_inventory(&txn);
            if let Some((bundle, fill, payload)) = &old_tail {
                assert_ne!(
                    txn.pool.ledger.table[txn.pool.ledger.table.len() - 2],
                    *bundle
                );
                assert_tail_payload(txn.pool, *bundle, *fill, payload)?;
            } else {
                assert!(txn.reservation.replaced_tail.is_none());
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
        for (page_tokens, page) in [
            (8, PageTokens::B8),
            (16, PageTokens::B16),
            (32, PageTokens::B32),
        ] {
            let max_context = page_tokens * 2 + 1;
            exercise_partial_tail_rollback(
                page_tokens,
                PagedKvPlan::new(geometry(max_context), page)?,
            )?;
            exercise_start_and_multi_page_append(
                page_tokens,
                PagedKvPlan::new(geometry(max_context), page)?,
                page_tokens,
                page_tokens + 1,
            )?;
            exercise_start_and_multi_page_append(
                page_tokens,
                PagedKvPlan::new(geometry(max_context), page)?,
                page_tokens + 1,
                page_tokens,
            )?;
        }
        Ok(())
    }

    #[test]
    fn all_page_sizes_support_prepared_cpu_commit_aggregates() -> Result<()> {
        const APPEND_TOKENS: usize = 2;

        for (page_tokens, page) in [
            (8, PageTokens::B8),
            (16, PageTokens::B16),
            (32, PageTokens::B32),
        ] {
            let start = page_tokens - 1;
            let plan = PagedKvPlan::new(geometry(page_tokens * 2 + 1), page)?;
            let mut first_pool = PagedKvPool::new(plan)?;
            let mut first_model = empty_model();
            seed_partial_tail(&mut first_pool, &mut first_model, start)?;
            let first_tail = tail_witness(&first_pool)?;

            {
                let mut incomplete = first_pool.begin_append(APPEND_TOKENS)?;
                write_layer(&mut incomplete, 0, start, APPEND_TOKENS)?;
                assert!(
                    matches!(
                        incomplete.prepare_commit(),
                        Err(Error::PagedIncompleteAppend { .. })
                    ),
                    "incomplete preparation must keep publication unpublished"
                );
            }
            assert_tail_restored(&first_pool, &first_model, &first_tail)?;

            {
                let prepared = prepare_all(&mut first_pool, start, APPEND_TOKENS)?;
                drop(prepared);
            }
            assert_tail_restored(&first_pool, &first_model, &first_tail)?;

            let mut second_pool = PagedKvPool::new(plan)?;
            let mut second_model = empty_model();
            seed_partial_tail(&mut second_pool, &mut second_model, start)?;
            let second_tail = tail_witness(&second_pool)?;

            let first_prepared = prepare_all(&mut first_pool, start, APPEND_TOKENS)?;
            {
                let mut incomplete = second_pool.begin_append(APPEND_TOKENS)?;
                write_layer(&mut incomplete, 0, start, APPEND_TOKENS)?;
                assert!(
                    matches!(
                        incomplete.prepare_commit(),
                        Err(Error::PagedIncompleteAppend { .. })
                    ),
                    "a second aggregate member must reject incomplete preparation"
                );
            }
            drop(first_prepared);
            assert_tail_restored(&first_pool, &first_model, &first_tail)?;
            assert_tail_restored(&second_pool, &second_model, &second_tail)?;

            let first_prepared = prepare_all(&mut first_pool, start, APPEND_TOKENS)?;
            let second_prepared = prepare_all(&mut second_pool, start, APPEND_TOKENS)?;
            first_prepared.commit();
            second_prepared.commit();
            append_model(&mut first_model, start, APPEND_TOKENS);
            append_model(&mut second_model, start, APPEND_TOKENS);
            assert_inventory(&first_pool, None);
            assert_inventory(&second_pool, None);
            assert_committed_matches(&first_pool, &first_model)?;
            assert_committed_matches(&second_pool, &second_model)?;
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn native_plan_uses_explicit_page_choice_and_one_spare_bundle() -> Result<()> {
        use kernels::attention::NativePageTokens;

        let geometry = PagedKvGeometry {
            layers: LAYERS,
            row_width: ROW_WIDTH,
            max_context: 20,
        };
        let native =
            NativePagedKvPlan::try_from_geometry(geometry, 1, ROW_WIDTH, NativePageTokens::B32)?;
        assert_eq!(native.layout().page_tokens(), 32);
        assert_eq!(native.layout().physical_pages(), 2);
        assert_eq!(
            native.layout().backing_elements(),
            LAYERS * 2 * 32 * ROW_WIDTH
        );
        assert_eq!(native.key_value_elements(), LAYERS * 2 * 32 * ROW_WIDTH);
        assert_eq!(native.page_table_elements(), 1);
        assert!(
            NativePagedKvPlan::try_from_geometry(geometry, 2, ROW_WIDTH, NativePageTokens::B8,)
                .is_err()
        );
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn native_attention_binding_rejects_same_row_width_with_different_gqa_axes()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use kernels::attention::{NativePageTokens, PagedDecodePlan};

        let cache = NativePagedKvPlan::try_from_geometry(
            PagedKvGeometry {
                layers: LAYERS,
                row_width: 8,
                max_context: 8,
            },
            1,
            8,
            NativePageTokens::B8,
        )?;
        let accepted = NativePagedDecodePlan::try_from_paged_decode(
            PagedDecodePlan::try_from_dimensions(1, 2, 1, 8)?,
            8,
            cache.layout().physical_pages(),
        )?;
        assert!(validate_native_attention_binding(cache, 1, accepted).is_ok());

        let remapped = NativePagedDecodePlan::try_from_paged_decode(
            PagedDecodePlan::try_from_dimensions(1, 2, 2, 4)?,
            8,
            cache.layout().physical_pages(),
        )?;
        assert!(matches!(
            validate_native_attention_binding(cache, 1, remapped),
            Err(Error::PagedLayout { .. })
        ));
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn native_stream_preflight_rejects_other_process_local_device() {
        assert!(ensure_same_process_device(3, 3).is_ok());
        assert!(matches!(
            ensure_same_process_device(3, 4),
            Err(Error::PagedNativeDeviceMismatch {
                expected: 3,
                actual: 4,
                ..
            })
        ));
    }

    #[test]
    fn prepared_native_commit_parks_one_verified_reservation_without_device_state() -> Result<()> {
        let plan = PagedKvPlan::new(geometry(8), PageTokens::B8)?;
        let mut ledger = PagedKvLedger::new(plan)?;
        let mut prepared = NativePreparedCommit::default();

        assert!(matches!(
            prepared.take(),
            Err(Error::PagedNativeCommitNotPrepared { .. })
        ));

        let incomplete = ledger.begin_append(1)?;
        let first = ledger.write_location(&incomplete, 0, 0)?;
        ledger.record_write(0, first.page, first.within)?;
        assert!(matches!(
            ledger.validate_commit(&incomplete),
            Err(Error::PagedIncompleteAppend { layer: 1, .. })
        ));
        ledger.rollback(&incomplete);

        let reservation = ledger.begin_append(1)?;
        record_all_ledger_rows(&mut ledger, &reservation)?;
        ledger.validate_commit(&reservation)?;
        prepared.park_verified(reservation);
        assert!(matches!(
            prepared.ensure_empty(),
            Err(Error::PagedNativeCommitPrepared { .. })
        ));
        assert_eq!(ledger.committed_tokens, 0);

        let mut reservation = prepared.take()?;
        ledger.publish_commit(&mut reservation);
        assert_eq!(ledger.committed_tokens, 1);
        assert!(matches!(
            prepared.take(),
            Err(Error::PagedNativeCommitNotPrepared { .. })
        ));
        Ok(())
    }
    #[test]
    fn boundary_capacity_and_shape_are_typed() -> Result<()> {
        let mut pool = PagedKvPool::new(PagedKvPlan::new(geometry(20), PageTokens::B8)?)?;
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
        let narrow_b8 = PagedKvPlan::new(narrow, PageTokens::B8)?;
        let narrow_b16 = PagedKvPlan::new(narrow, PageTokens::B16)?;
        let narrow_b32 = PagedKvPlan::new(narrow, PageTokens::B32)?;
        assert_plan_request(narrow_b8, narrow, 8);
        assert_plan_request(narrow_b16, narrow, 16);
        assert_plan_request(narrow_b32, narrow, 32);
        assert_eq!(PagedKvPlan::select(narrow)?.page_tokens, PageTokens::B16);
        let wide = PagedKvGeometry {
            layers: 1,
            row_width: 64,
            max_context: 128,
        };
        let wide_b8 = PagedKvPlan::new(wide, PageTokens::B8)?;
        let wide_b16 = PagedKvPlan::new(wide, PageTokens::B16)?;
        assert_plan_request(wide_b8, wide, 8);
        assert_plan_request(wide_b16, wide, 16);
        assert_eq!(PagedKvPlan::select(wide)?.page_tokens, PageTokens::B8);
        assert!(matches!(
            PagedKvPlan::new(
                PagedKvGeometry {
                    layers: usize::MAX,
                    row_width: 2,
                    max_context: 1
                },
                PageTokens::B8
            ),
            Err(Error::PagedArithmetic { .. })
        ));
        Ok(())
    }

    fn assert_plan_request(plan: PagedKvPlan, geometry: PagedKvGeometry, page_tokens: usize) {
        let page_count = geometry.max_context.div_ceil(page_tokens);
        let bundle_count = page_count + 1;
        let requested_f32 = bundle_count * geometry.layers * 2 * page_tokens * geometry.row_width;
        let metadata_bytes = (page_count * 2 + bundle_count + geometry.layers) * size_of::<usize>();
        let total_requested_bytes = requested_f32 * size_of::<f32>() + metadata_bytes;
        assert_eq!(plan.requested_f32_elements(), requested_f32);
        assert_eq!(plan.allocation.total_requested_bytes, total_requested_bytes);
    }
}
