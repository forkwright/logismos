//! Safe `Device` wrapper over the HIP runtime API.

use std::ffi::c_int;
use std::sync::Arc;

use crate::error::{Error, Result, check};
use crate::ffi;

/// Device ordinal. Stable across runs; `0` is the first reported device.
pub(crate) type DeviceOrdinal = c_int;

/// PCI bus identifier in `domain:bus:device.function` form, e.g. `0000:03:00.0`.
///
/// Correlates a HIP device with `rocm-smi` output. Opaque: the runtime reports
/// the three numeric components separately and this type is the assembled form.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PciBusId(String);

impl PciBusId {
    /// Wraps an already-formatted `domain:bus:device.function` string.
    #[must_use]
    pub const fn new(id: String) -> Self {
        Self(id)
    }

    /// Borrows the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PciBusId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Static properties of a HIP device (a subset of the full
/// `hipDeviceProp_t`).
#[derive(Clone, Debug)]
pub struct DeviceProps {
    /// ISA reported by the runtime, e.g. `"gfx1100"`.
    pub isa: String,
    /// Marketing name, e.g. `"AMD Radeon Pro W7900"`.
    pub name: String,
    /// Total VRAM in bytes.
    pub total_vram_bytes: u64,
    /// Number of compute units on the device.
    ///
    /// On RDNA3 HIP reports this as the Workgroup-Processor count
    /// (48 on the W7900), which is half the physical CU count (96).
    /// Treat this as a workgroup-scheduling unit, not a direct
    /// throughput factor.
    pub compute_units: u32,
    /// Native wavefront size (32 on RDNA3, 64 on CDNA).
    pub wavefront_size: u32,
    /// Maximum threads per workgroup.
    pub max_threads_per_block: u32,
    /// LDS bytes visible to a single workgroup (group segment).
    pub max_shared_mem_per_block: u32,
    /// Maximum core clock reported by the runtime, in kHz.
    pub clock_rate_khz: u32,
    /// PCI bus identifier (for correlation with `rocm-smi`).
    pub pci_bus_id: PciBusId,
}

/// Free / total VRAM snapshot.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBudget {
    /// Free VRAM in bytes at the time of the call.
    pub free: u64,
    /// Total VRAM in bytes.
    pub total: u64,
}

/// Safe handle to a HIP device.
///
/// `Device` is cheap to clone — it holds an `Arc` to the immutable
/// per-device state. Every allocation, stream, and kernel launch in
/// logismos must be associated with a `Device`.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    ordinal: DeviceOrdinal,
    props: DeviceProps,
}

impl Device {
    /// Open the device at `ordinal`.
    ///
    /// # Errors
    ///
    /// - [`Error::NoSuchDevice`] if `ordinal >= device count`.
    /// - [`Error::Runtime`] for any HIP API failure.
    pub fn new(ordinal: DeviceOrdinal) -> Result<Self> {
        let mut count: c_int = 0;
        // SAFETY: FFI call; `&mut count` is a valid pointer.
        check(
            unsafe { ffi::hipGetDeviceCount(&mut count) },
            "hipGetDeviceCount",
        )?;
        if ordinal < 0 || ordinal >= count {
            return Err(Error::NoSuchDevice { ordinal, count });
        }
        // SAFETY: ordinal is bounds-checked above.
        check(unsafe { ffi::hipSetDevice(ordinal) }, "hipSetDevice")?;

        let mut raw: ffi::hipDeviceProp_t = Default::default();
        // SAFETY: `&mut raw` is valid; ordinal bounds-checked.
        check(
            unsafe { ffi::hipGetDeviceProperties(&mut raw, ordinal) },
            "hipGetDeviceProperties",
        )?;

        let props = DeviceProps::from_raw(&raw)?;
        Ok(Self {
            inner: Arc::new(DeviceInner { ordinal, props }),
        })
    }

    /// Builds a `Device` carrying an out-of-range ordinal, bypassing the
    /// `hipGetDeviceCount` bounds check in [`Device::new`].
    ///
    /// WHY: exists so tests elsewhere in this crate can drive a real,
    /// deterministic `make_current` (and therefore `hipSetDevice`)
    /// failure without a physical GPU. An out-of-range ordinal is
    /// `hipErrorInvalidDevice` per the HIP API contract regardless of
    /// how many real devices are present, including zero — unlike a
    /// genuine hardware fault, this failure mode needs no device at
    /// all, only the runtime library CI already links against.
    #[cfg(test)]
    pub(crate) fn invalid_for_test() -> Self {
        Self {
            inner: Arc::new(DeviceInner {
                ordinal: DeviceOrdinal::MAX,
                props: DeviceProps {
                    isa: String::new(),
                    name: String::new(),
                    total_vram_bytes: 0,
                    compute_units: 0,
                    wavefront_size: 0,
                    max_threads_per_block: 0,
                    max_shared_mem_per_block: 0,
                    clock_rate_khz: 0,
                    pci_bus_id: PciBusId::new(String::new()),
                },
            }),
        }
    }

    /// Device ordinal.
    #[must_use]
    pub fn ordinal(&self) -> DeviceOrdinal {
        self.inner.ordinal
    }

    /// Immutable reference to the cached device properties.
    #[must_use]
    pub fn props(&self) -> &DeviceProps {
        &self.inner.props
    }

    /// Number of live `Device` handles (clones of this value, or clones
    /// held inside another type such as `DeviceBuffer` or `Stream`)
    /// sharing the same underlying device state.
    ///
    /// WHY: exists so tests elsewhere in this crate can observe whether
    /// a value holding a cloned `Device` (e.g. `memory::PendingCopy`'s
    /// wrapped `DeviceBuffer`) was actually dropped, without needing
    /// real HIP device state to exercise the drop path end-to-end — a
    /// dropped clone decrements this count, an un-dropped (leaked) one
    /// does not, and that difference is observable with no hardware.
    #[cfg(test)]
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Make this device current on the calling thread.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] on HIP failure.
    pub(crate) fn make_current(&self) -> Result<()> {
        // SAFETY: FFI call; ordinal validated at construction.
        check(
            unsafe { ffi::hipSetDevice(self.inner.ordinal) },
            "hipSetDevice",
        )
    }

    /// Snapshot the device's free / total VRAM via `hipMemGetInfo`.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] on HIP failure.
    pub(crate) fn memory_budget(&self) -> Result<MemoryBudget> {
        self.make_current()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: both pointers valid for the duration of the call.
        check(
            unsafe { ffi::hipMemGetInfo(&mut free, &mut total) },
            "hipMemGetInfo",
        )?;
        Ok(MemoryBudget {
            free: u64::try_from(free).unwrap_or(u64::MAX),
            total: u64::try_from(total).unwrap_or(u64::MAX),
        })
    }

    /// Block the current thread until all outstanding work on this
    /// device's default stream completes.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] on HIP failure.
    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        // SAFETY: FFI call; device is current.
        check(
            unsafe { ffi::hipDeviceSynchronize() },
            "hipDeviceSynchronize",
        )
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("ordinal", &self.inner.ordinal)
            .field("isa", &self.inner.props.isa)
            .field("name", &self.inner.props.name)
            .field("vram_bytes", &self.inner.props.total_vram_bytes)
            .finish()
    }
}

impl DeviceProps {
    fn from_raw(raw: &ffi::hipDeviceProp_t) -> Result<Self> {
        // ISA and name arrive as null-terminated C char arrays.
        let isa = cstr_from_array(&raw.gcnArchName);
        let name = cstr_from_array(&raw.name);
        let pci_domain = device_prop_u32("pciDomainID", raw.pciDomainID)?;
        let pci_bus = device_prop_u32("pciBusID", raw.pciBusID)?;
        let pci_device = device_prop_u32("pciDeviceID", raw.pciDeviceID)?;
        let pci = format!("{pci_domain:04x}:{pci_bus:02x}:{pci_device:02x}.0");
        Ok(Self {
            isa,
            name,
            total_vram_bytes: u64::try_from(raw.totalGlobalMem).unwrap_or(u64::MAX),
            compute_units: device_prop_u32("multiProcessorCount", raw.multiProcessorCount)?,
            wavefront_size: device_prop_u32("warpSize", raw.warpSize)?,
            max_threads_per_block: device_prop_u32("maxThreadsPerBlock", raw.maxThreadsPerBlock)?,
            max_shared_mem_per_block: device_prop_u32("sharedMemPerBlock", raw.sharedMemPerBlock)?,
            clock_rate_khz: device_prop_u32("clockRate", raw.clockRate)?,
            pci_bus_id: PciBusId::new(pci),
        })
    }
}

fn device_prop_u32<T>(field: &'static str, value: T) -> Result<u32>
where
    u32: TryFrom<T>,
    <u32 as TryFrom<T>>::Error: std::fmt::Display,
{
    u32::try_from(value).map_err(|err| {
        Error::Internal(format!(
            "HIP device property `{field}` is outside the u32 range: {err}"
        ))
    })
}

fn cstr_from_array(bytes: &[i8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let slice: &[u8] = unsafe {
        // SAFETY: `bytes` lives for the duration of the call and the
        // layout of `i8` and `u8` is identical.
        core::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), end)
    };
    String::from_utf8_lossy(slice).into_owned()
}

/// Returns the number of HIP devices visible to the runtime.
///
/// # Errors
///
/// [`Error::Runtime`] on HIP failure.
pub fn device_count() -> Result<i32> {
    let mut count: c_int = 0;
    // SAFETY: `&mut count` is valid.
    check(
        unsafe { ffi::hipGetDeviceCount(&mut count) },
        "hipGetDeviceCount",
    )?;
    Ok(count)
}
