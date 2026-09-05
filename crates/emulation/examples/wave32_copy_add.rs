//! Runs the LLVM-verified raw copy-and-add fixture on CPU wave32 state.

use emulation::Wave32Program;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = vec![
        0x00, 0x03, 0x02, 0x7e, // v_mov_b32_e32 v1, v0
        0x00, 0x03, 0x04, 0x4a, // v_add_nc_u32_e32 v2, v0, v1
        0x00, 0x00, 0xb0, 0xbf, // s_endpgm
    ];
    let mut source = [0u32; 32];
    for (lane, value) in source.iter_mut().enumerate() {
        *value = u32::from(lane.to_le_bytes()[0]);
    }

    let report = Wave32Program::new(text, vec![source, [0; 32], [0; 32]], 3)?.execute()?;
    println!(
        "lane 0 copy={} sum={}",
        report.registers()[1][0],
        report.registers()[2][0]
    );
    Ok(())
}
