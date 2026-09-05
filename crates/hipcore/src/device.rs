//! Safe device discovery and selection over the HIP runtime API.

use std::ffi::c_int;
use std::fmt;
use std::sync::Arc;

use crate::error::{
    InternalSnafu, NoDeviceWithPciBusIdSnafu, NoDeviceWithUuidSnafu, NoSuchDeviceSnafu, Result,
    UnsupportedIsaSnafu, check,
};
use crate::ffi;

fn hip_target() -> &'static str {
    include_str!("../../../contracts/gpu-target.txt").trim()
}

/// Device ordinal assigned by HIP for the current process visibility set.
///
/// This value is suitable for immediate HIP calls, but is not a stable hardware
/// identity: visibility masks and device ordering can change it between runs.
pub(crate) type DeviceOrdinal = c_int;

/// PCI bus identifier in `domain:bus:device.function` form, e.g. `0000:03:00.0`.
///
/// Correlates a HIP device with topology tooling. It is a stable fallback when
/// HIP does not report a usable UUID.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PciBusId(String);

impl PciBusId {
    /// Wrap an already-formatted PCI bus identifier.
    #[must_use]
    pub const fn new(id: String) -> Self {
        Self(id)
    }

    /// Borrow the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PciBusId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// HIP-provided 16-byte device UUID.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceUuid([u8; 16]);

impl DeviceUuid {
    /// Construct a UUID from its HIP byte representation.
    #[must_use]
    pub const fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Borrow the UUID bytes in HIP order.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl fmt::Display for DeviceUuid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, byte) in self.0.iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                formatter.write_str("-")?;
            }
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Stable hardware identity for a HIP device.
///
/// UUID is preferred because it follows the device across PCI topology changes.
/// HIP occasionally reports an all-zero UUID; in that case selection falls back
/// to the PCI bus identifier instead of treating an unusable UUID as identity.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DeviceIdentity {
    uuid: Option<DeviceUuid>,
    pci_bus_id: PciBusId,
}

impl DeviceIdentity {
    /// Construct a stable identity from HIP properties.
    #[must_use]
    pub const fn new(uuid: Option<DeviceUuid>, pci_bus_id: PciBusId) -> Self {
        Self { uuid, pci_bus_id }
    }

    /// Return the UUID when HIP reported a non-zero value.
    #[must_use]
    pub const fn uuid(&self) -> Option<DeviceUuid> {
        self.uuid
    }

    /// Return the PCI topology identifier.
    #[must_use]
    pub fn pci_bus_id(&self) -> &PciBusId {
        &self.pci_bus_id
    }

    /// Select this identity by UUID when available, otherwise by PCI address.
    #[must_use]
    pub fn preferred_selector(&self) -> DeviceSelector {
        match self.uuid {
            Some(uuid) => DeviceSelector::Uuid(uuid),
            None => DeviceSelector::PciBusId(self.pci_bus_id.clone()),
        }
    }
}

/// A bounded selector for a device visible to the current HIP process.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DeviceSelector {
    /// Current-process HIP ordinal. This is not stable across visibility changes.
    Ordinal(c_int),
    /// Stable HIP UUID.
    Uuid(DeviceUuid),
    /// Stable PCI topology fallback when UUID is unavailable.
    PciBusId(PciBusId),
}

impl DeviceSelector {
    fn matches(&self, info: &DeviceInfo) -> bool {
        match self {
            Self::Ordinal(ordinal) => info.ordinal == *ordinal,
            Self::Uuid(uuid) => info.identity().uuid() == Some(*uuid),
            Self::PciBusId(pci_bus_id) => info.identity().pci_bus_id() == pci_bus_id,
        }
    }

    fn not_found<T>(&self, count: c_int) -> Result<T> {
        match self {
            Self::Ordinal(ordinal) => NoSuchDeviceSnafu {
                ordinal: *ordinal,
                count,
            }
            .fail(),
            Self::Uuid(uuid) => NoDeviceWithUuidSnafu { uuid: *uuid }.fail(),
            Self::PciBusId(pci_bus_id) => NoDeviceWithPciBusIdSnafu {
                pci_bus_id: pci_bus_id.clone(),
            }
            .fail(),
        }
    }
}

/// Static properties of a HIP device (a subset of the full `hipDeviceProp_t`).
#[derive(Clone, Debug)]
pub struct DeviceProps {
    /// ISA reported by the runtime, e.g. `"gfx1100"`.
    pub isa: String,
    /// Marketing name reported by the runtime.
    pub name: String,
    /// Total VRAM in bytes.
    pub total_vram_bytes: u64,
    /// Number of compute units reported by the runtime.
    pub compute_units: u32,
    /// Native wavefront size (32 on RDNA3, 64 on CDNA).
    pub wavefront_size: u32,
    /// Maximum threads per workgroup.
    pub max_threads_per_block: u32,
    /// LDS bytes visible to a single workgroup (group segment).
    pub max_shared_mem_per_block: u32,
    /// Maximum core clock reported by the runtime, in kHz.
    pub clock_rate_khz: u32,
    identity: DeviceIdentity,
}

impl DeviceProps {
    /// Return the stable hardware identity reported by HIP.
    #[must_use]
    pub fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    /// Whether this ISA is exactly the configured target, with only valid HIP
    /// feature suffixes permitted after the architecture name.
    #[must_use]
    pub fn supports_target(&self) -> bool {
        isa_matches_target(&self.isa)
    }

    fn require_target(&self) -> Result<()> {
        if self.supports_target() {
            Ok(())
        } else {
            UnsupportedIsaSnafu {
                isa: self.isa.clone(),
            }
            .fail()
        }
    }

    fn from_raw(raw: &ffi::hipDeviceProp_t) -> Result<Self> {
        let pci_domain = device_prop_u32("pciDomainID", raw.pciDomainID)?;
        let pci_bus = device_prop_u32("pciBusID", raw.pciBusID)?;
        let pci_device = device_prop_u32("pciDeviceID", raw.pciDeviceID)?;
        let pci_bus_id =
            PciBusId::new(format!("{pci_domain:04x}:{pci_bus:02x}:{pci_device:02x}.0"));
        let uuid_bytes = raw.uuid.bytes.map(|byte| byte as u8);
        let uuid =
            (!uuid_bytes.iter().all(|byte| *byte == 0)).then_some(DeviceUuid::new(uuid_bytes));
        Ok(Self {
            isa: cstr_from_array(&raw.gcnArchName),
            name: cstr_from_array(&raw.name),
            total_vram_bytes: u64::try_from(raw.totalGlobalMem).unwrap_or(u64::MAX),
            compute_units: device_prop_u32("multiProcessorCount", raw.multiProcessorCount)?,
            wavefront_size: device_prop_u32("warpSize", raw.warpSize)?,
            max_threads_per_block: device_prop_u32("maxThreadsPerBlock", raw.maxThreadsPerBlock)?,
            max_shared_mem_per_block: device_prop_u32("sharedMemPerBlock", raw.sharedMemPerBlock)?,
            clock_rate_khz: device_prop_u32("clockRate", raw.clockRate)?,
            identity: DeviceIdentity::new(uuid, pci_bus_id),
        })
    }
}

/// A device discovered in the current HIP visibility set.
///
/// The ordinal is a transient handle for this process; use [`DeviceInfo::identity`]
/// when persisting or comparing device selection across runs.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    ordinal: DeviceOrdinal,
    props: DeviceProps,
}

impl DeviceInfo {
    /// Return the current-process ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> DeviceOrdinal {
        self.ordinal
    }

    /// Return immutable static properties.
    #[must_use]
    pub fn props(&self) -> &DeviceProps {
        &self.props
    }

    /// Return the stable hardware identity.
    #[must_use]
    pub fn identity(&self) -> &DeviceIdentity {
        self.props.identity()
    }
}

/// Free / total VRAM snapshot.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBudget {
    /// Free VRAM in bytes at the time of the call.
    pub free: u64,
    /// Total VRAM in bytes.
    pub total: u64,
}

/// Safe handle to a supported HIP device.
///
/// `Device` is cheap to clone — it holds an `Arc` to immutable per-device
/// state. Construction rejects an unsupported ISA before making the device
/// current, allocating memory, creating streams, or dispatching work.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    ordinal: DeviceOrdinal,
    props: DeviceProps,
}

impl Device {
    /// Open the visible device at `ordinal`.
    ///
    /// The ordinal is process-local and can change after a reboot or visibility
    /// mask change. Use [`Device::select`] with a stable selector when that
    /// distinction matters.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::NoSuchDevice`] if `ordinal` is not visible.
    /// - [`crate::Error::UnsupportedIsa`] if the device ISA cannot execute this build's target.
    /// - [`crate::Error::Runtime`] for HIP API failures.
    pub fn new(ordinal: DeviceOrdinal) -> Result<Self> {
        let info = query_device(ordinal)?;
        Self::open_info(info)
    }

    /// Open a visible device selected by its ordinal, UUID, or PCI address.
    ///
    /// # Errors
    ///
    /// Returns a selector-specific missing-device error when no visible device
    /// matches, [`crate::Error::UnsupportedIsa`] before HIP activation for an
    /// incompatible device, or [`crate::Error::Runtime`] for HIP failures.
    pub fn select(selector: &DeviceSelector) -> Result<Self> {
        Self::select_optional(selector)?.map_or_else(|| selector.not_found(device_count()?), Ok)
    }

    /// Open a visible device when one matches `selector`.
    ///
    /// This is the explicit optional-selection form: absence is `Ok(None)`,
    /// while HIP and ISA failures remain errors.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedIsa`] before HIP activation for a
    /// matched incompatible device, or [`crate::Error::Runtime`] for HIP failures.
    pub fn select_optional(selector: &DeviceSelector) -> Result<Option<Self>> {
        let devices = enumerate_devices()?;
        let Some(info) = devices.into_iter().find(|info| selector.matches(info)) else {
            return Ok(None);
        };
        Self::open_info(info).map(Some)
    }

    fn open_info(info: DeviceInfo) -> Result<Self> {
        info.props.require_target()?;
        // SAFETY: the ordinal originated from a bounded HIP device query.
        check(unsafe { ffi::hipSetDevice(info.ordinal) }, "hipSetDevice")?;
        Ok(Self {
            inner: Arc::new(DeviceInner {
                ordinal: info.ordinal,
                props: info.props,
            }),
        })
    }

    /// Builds a `Device` carrying an out-of-range ordinal, bypassing HIP discovery.
    ///
    /// WHY: crate-local lifecycle tests need a deterministic `hipSetDevice`
    /// failure without querying or requiring a physical GPU.
    #[cfg(test)]
    pub(crate) fn invalid_for_test() -> Self {
        Self {
            inner: Arc::new(DeviceInner {
                ordinal: DeviceOrdinal::MAX,
                props: test_props(),
            }),
        }
    }

    /// Return the transient HIP ordinal.
    #[must_use]
    pub fn ordinal(&self) -> DeviceOrdinal {
        self.inner.ordinal
    }

    /// Return cached immutable device properties.
    #[must_use]
    pub fn props(&self) -> &DeviceProps {
        &self.inner.props
    }

    /// Number of live cloned handles sharing this device state.
    #[cfg(test)]
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Make this device current on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Runtime`] on HIP failure.
    pub(crate) fn make_current(&self) -> Result<()> {
        // SAFETY: construction validated the ISA and bounded the ordinal.
        check(
            unsafe { ffi::hipSetDevice(self.inner.ordinal) },
            "hipSetDevice",
        )
    }

    /// Snapshot free and total VRAM via `hipMemGetInfo`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Runtime`] on HIP failure.
    pub(crate) fn memory_budget(&self) -> Result<MemoryBudget> {
        self.make_current()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: both pointers are valid for the duration of this call.
        check(
            unsafe { ffi::hipMemGetInfo(&mut free, &mut total) },
            "hipMemGetInfo",
        )?;
        Ok(MemoryBudget {
            free: u64::try_from(free).unwrap_or(u64::MAX),
            total: u64::try_from(total).unwrap_or(u64::MAX),
        })
    }

    /// Block until outstanding work on this device's default stream completes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Runtime`] on HIP failure.
    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        // SAFETY: the device is current on this thread.
        check(
            unsafe { ffi::hipDeviceSynchronize() },
            "hipDeviceSynchronize",
        )
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Device")
            .field("ordinal", &self.inner.ordinal)
            .field("isa", &self.inner.props.isa)
            .field("identity", &self.inner.props.identity)
            .finish()
    }
}

/// Enumerate every device visible to the current HIP process.
///
/// This inspects properties only; it does not make a device current, allocate
/// memory, create a stream, or dispatch GPU work. Unsupported devices remain
/// visible so callers receive an ISA error rather than an ambiguous absence.
///
/// # Errors
///
/// Returns [`crate::Error::Runtime`] for HIP discovery or property-query failures.
pub fn enumerate_devices() -> Result<Vec<DeviceInfo>> {
    let count = device_count()?;
    (0..count).map(query_device).collect()
}

fn query_device(ordinal: DeviceOrdinal) -> Result<DeviceInfo> {
    let count = device_count()?;
    if ordinal < 0 || ordinal >= count {
        return NoSuchDeviceSnafu { ordinal, count }.fail();
    }
    let mut raw: ffi::hipDeviceProp_t = Default::default();
    // SAFETY: `raw` is valid output storage and ordinal was bounds-checked.
    check(
        unsafe { ffi::hipGetDeviceProperties(&mut raw, ordinal) },
        "hipGetDeviceProperties",
    )?;
    Ok(DeviceInfo {
        ordinal,
        props: DeviceProps::from_raw(&raw)?,
    })
}

fn isa_matches_target(isa: &str) -> bool {
    let Some((architecture, suffixes)) = isa.split_once(':') else {
        return isa == hip_target();
    };
    architecture == hip_target() && suffixes.split(':').all(valid_isa_feature)
}

fn valid_isa_feature(feature: &str) -> bool {
    let Some((name, state)) = feature.split_at_checked(feature.len().saturating_sub(1)) else {
        return false;
    };
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        && matches!(state, "+" | "-")
}

fn device_prop_u32<T>(field: &'static str, value: T) -> Result<u32>
where
    u32: TryFrom<T>,
    <u32 as TryFrom<T>>::Error: fmt::Display,
{
    u32::try_from(value).map_err(|error| {
        InternalSnafu {
            message: format!("HIP device property `{field}` is outside the u32 range: {error}"),
        }
        .build()
    })
}

fn cstr_from_array(bytes: &[i8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    let slice: &[u8] = unsafe {
        // SAFETY: i8 and u8 have identical layout; `bytes` remains live.
        core::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), end)
    };
    String::from_utf8_lossy(slice).into_owned()
}

/// Return the number of HIP devices visible to the runtime.
///
/// # Errors
///
/// Returns [`crate::Error::Runtime`] on HIP failure.
pub fn device_count() -> Result<i32> {
    let mut count: c_int = 0;
    // SAFETY: `count` is valid output storage.
    check(
        unsafe { ffi::hipGetDeviceCount(&mut count) },
        "hipGetDeviceCount",
    )?;
    Ok(count)
}

#[cfg(test)]
fn test_props() -> DeviceProps {
    DeviceProps {
        isa: hip_target().to_string(),
        name: String::new(),
        total_vram_bytes: 0,
        compute_units: 0,
        wavefront_size: 0,
        max_threads_per_block: 0,
        max_shared_mem_per_block: 0,
        clock_rate_khz: 0,
        identity: DeviceIdentity::new(None, PciBusId::new(String::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn fixture(
        ordinal: c_int,
        uuid: Option<DeviceUuid>,
        pci_bus_id: &str,
        vram_gib: u64,
    ) -> DeviceInfo {
        DeviceInfo {
            ordinal,
            props: DeviceProps {
                isa: hip_target().to_string(),
                name: "fixture".to_string(),
                total_vram_bytes: vram_gib * GIB,
                compute_units: 1,
                wavefront_size: 32,
                max_threads_per_block: 1,
                max_shared_mem_per_block: 1,
                clock_rate_khz: 1,
                identity: DeviceIdentity::new(uuid, PciBusId::new(pci_bus_id.to_string())),
            },
        }
    }

    #[test]
    fn target_isa_accepts_valid_feature_suffixes() {
        assert!(
            isa_matches_target(hip_target()),
            "bare target ISA must be accepted"
        );
        assert!(
            isa_matches_target(&format!("{}:sramecc+:xnack-", hip_target())),
            "valid HIP feature suffixes must be accepted"
        );
    }

    #[test]
    fn target_isa_rejects_lookalikes_and_malformed_suffixes() {
        assert!(
            !isa_matches_target("gfx11000"),
            "lookalike ISA must not match"
        );
        assert!(
            !isa_matches_target(&format!("{}:xnack", hip_target())),
            "feature suffix must include its state"
        );
        assert!(
            !isa_matches_target(&format!("{}::xnack+", hip_target())),
            "empty feature suffix must not match"
        );
    }

    #[test]
    fn uuid_selection_survives_ordinal_renumbering() {
        let uuid = DeviceUuid::new([7; 16]);
        let first_visibility = [fixture(0, Some(uuid), "0000:01:00.0", 48)];
        let renumbered_visibility = [
            fixture(0, None, "0000:02:00.0", 24),
            fixture(1, Some(uuid), "0000:01:00.0", 48),
        ];
        let selector = first_visibility[0].identity().preferred_selector();
        assert_eq!(
            renumbered_visibility
                .iter()
                .find(|info| selector.matches(info))
                .map(DeviceInfo::ordinal),
            Some(1),
            "UUID selection must find the same device after ordinal renumbering"
        );
    }

    #[test]
    fn all_zero_uuid_falls_back_to_pci_selection() {
        let device = fixture(3, None, "0000:03:00.0", 24);
        assert_eq!(
            device.identity().preferred_selector(),
            DeviceSelector::PciBusId(PciBusId::new("0000:03:00.0".to_string())),
            "missing UUID must select by PCI topology"
        );
    }

    #[test]
    fn target_support_does_not_depend_on_vram_capacity_or_marketing_name() {
        let workstation = fixture(0, Some(DeviceUuid::new([1; 16])), "0000:01:00.0", 48);
        let consumer = fixture(1, Some(DeviceUuid::new([2; 16])), "0000:02:00.0", 24);
        assert!(
            workstation.props().supports_target(),
            "48 GiB fixture must be accepted"
        );
        assert!(
            consumer.props().supports_target(),
            "24 GiB fixture must be accepted"
        );
    }

    #[test]
    fn absent_requested_and_optional_devices_are_distinct() {
        let devices = [fixture(
            0,
            Some(DeviceUuid::new([1; 16])),
            "0000:01:00.0",
            24,
        )];
        let missing = DeviceSelector::Uuid(DeviceUuid::new([9; 16]));
        assert!(
            devices.iter().all(|info| !missing.matches(info)),
            "requested device fixture must be absent"
        );
        let optional = devices.iter().find(|info| missing.matches(info));
        assert!(
            optional.is_none(),
            "optional selection must resolve to no device"
        );
        assert!(
            matches!(missing.not_found::<()>(1), Err(crate::Error::NoDeviceWithUuid { .. })),
            "required UUID selection must report the UUID-specific missing-device error"
        );
    }
}
