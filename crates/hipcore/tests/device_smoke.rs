//! On-device smoke: explicit physical-HIP qualification only.

#![expect(
    clippy::expect_used,
    clippy::cast_precision_loss,
    reason = "HIP smoke tests use direct assertions and casts against device properties"
)]

use hipcore::{Device, DeviceBuffer, Stream};

#[test]
#[ignore = "requires an explicitly-invoked physical HIP device qualification"]
fn device_properties_match_target_capability() {
    let d = Device::new(0).expect("open device 0");
    let p = d.props();
    assert!(
        p.supports_target(),
        "device ISA `{}` does not match the configured HIP target",
        p.isa
    );
    assert_eq!(p.wavefront_size, 32, "target capability requires wave32");
}

#[test]
#[ignore = "requires an explicitly-invoked physical HIP device qualification"]
fn roundtrip_host_device_host() {
    let d = Device::new(0).expect("open");
    let src: Vec<f32> = (0..1024).map(|i| i as f32 / 7.0).collect();
    let buf = DeviceBuffer::from_host(&d, &src).expect("alloc+h2d");
    let mut dst = vec![0.0f32; src.len()];
    buf.copy_to_host(&mut dst).expect("d2h");
    assert_eq!(src, dst);
}

#[test]
#[ignore = "requires an explicitly-invoked physical HIP device qualification"]
fn stream_create_and_synchronize() {
    let d = Device::new(0).expect("open");
    let s = Stream::new(&d).expect("stream");
    s.synchronize().expect("sync");
}
