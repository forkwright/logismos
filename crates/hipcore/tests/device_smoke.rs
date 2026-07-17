//! On-device smoke: requires a functioning W7900 (gfx1100) on menos.

#![expect(
    clippy::expect_used,
    clippy::cast_precision_loss,
    reason = "HIP smoke tests use direct assertions and casts against device properties"
)]

use hipcore::{Device, DeviceBuffer, Stream};

fn have_gpu() -> bool {
    matches!(hipcore::device::device_count(), Ok(c) if c > 0)
}

#[test]
fn device_properties_match_w7900() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device visible");
        return;
    }
    let d = Device::new(0).expect("open device 0");
    let p = d.props();
    // Hardware assertions per PLAN.md §4 exit criterion 2.
    assert_eq!(p.isa, "gfx1100", "expected gfx1100, got {}", p.isa);
    assert_eq!(p.wavefront_size, 32);
    // HIP reports WGPs (48) on RDNA3, not CUs (96). Accept either.
    assert!(
        p.compute_units >= 48,
        "compute_units {} below WGP floor of 48",
        p.compute_units
    );
    assert!(
        p.total_vram_bytes >= 40 * (1 << 30),
        "vram {} below 40 GiB floor",
        p.total_vram_bytes
    );
}

#[test]
fn roundtrip_host_device_host() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device visible");
        return;
    }
    let d = Device::new(0).expect("open");
    let src: Vec<f32> = (0..1024).map(|i| i as f32 / 7.0).collect();
    let buf = DeviceBuffer::from_host(&d, &src).expect("alloc+h2d");
    let mut dst = vec![0.0f32; src.len()];
    buf.copy_to_host(&mut dst).expect("d2h");
    assert_eq!(src, dst);
}

#[test]
fn stream_create_and_synchronize() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device visible");
        return;
    }
    let d = Device::new(0).expect("open");
    let s = Stream::new(&d).expect("stream");
    s.synchronize().expect("sync");
}
