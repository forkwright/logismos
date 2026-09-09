//! `kernels` build script.
//!
//! Compiles every `.hip` source under `src/**/hip/*.hip` plus the
//! matching `_launcher.cpp` shim, and links them together into a
//! single static archive that Rust consumes via
//! `cargo:rustc-link-lib=static=logismos_kernels`.
//!
//! `LOGISMOS_HIP_BUILD=required` compiles those sources with `hipcc`
//! (ROCm ≥ 6.4) and fails if the compiler is absent. `cpu-only` builds
//! the CPU references only; the launchers return [`Error::NoGpuBuild`].

#![expect(
    clippy::doc_markdown,
    reason = "build-script docs use ROCm and cargo cfg spelling that trip doc_markdown"
)]

use std::env;
use std::path::{Path, PathBuf};
// kanon:ignore RUST/no-direct-process-command -- a build script runs before the workspace is built, so no project process wrapper is linkable here
use std::process::Command;

const HIP_BUILD_MODE_ENV: &str = "LOGISMOS_HIP_BUILD";
const HIP_BUILD_REQUIRED: &str = "required";
const HIP_BUILD_CPU_ONLY: &str = "cpu-only";
const HIP_TARGET: &str = include_str!("../../contracts/gpu-target.txt").trim_ascii();

enum HipBuildMode {
    Required,
    CpuOnly,
}

struct HipBuildConfiguration {
    mode: HipBuildMode,
    explicitly_set: bool,
}

fn main() -> Result<(), String> {
    println!("cargo:rustc-check-cfg=cfg(logismos_no_gpu_kernels)");
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run on any .hip or .cpp change under src/.
    for entry in walk_sources(&PathBuf::from("src")) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed={HIP_BUILD_MODE_ENV}");
    println!("cargo:rerun-if-env-changed=LOGISMOS_SKIP_HIP_BUILD");
    println!("cargo:rerun-if-changed=../../contracts/gpu-target.txt");

    if env::var("LOGISMOS_SKIP_HIP_BUILD").is_ok() {
        return Err(
            "LOGISMOS_SKIP_HIP_BUILD is retired; set LOGISMOS_HIP_BUILD=cpu-only instead"
                .to_string(),
        );
    }

    let hip_build = hip_build_mode()?;
    if !gpu_feature_enabled() {
        if hip_build.explicitly_set && matches!(hip_build.mode, HipBuildMode::Required) {
            return Err(
                "kernels/gpu is disabled but LOGISMOS_HIP_BUILD=required explicitly requests HIP compilation; enable the gpu feature or set LOGISMOS_HIP_BUILD=cpu-only"
                    .to_string(),
            );
        }
        return Ok(());
    }

    let out_dir = match env::var("OUT_DIR") {
        Ok(directory) => PathBuf::from(directory),
        Err(error) => return Err(format!("OUT_DIR is set by cargo: {error}")),
    };
    if matches!(hip_build.mode, HipBuildMode::CpuOnly) {
        println!("cargo:warning=HIP kernel compile disabled (LOGISMOS_HIP_BUILD=cpu-only)");
        println!("cargo:rustc-cfg=logismos_no_gpu_kernels");
        return Ok(());
    }

    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    if which(&hipcc).is_none() {
        return Err(format!(
            "hipcc not found on PATH while {HIP_BUILD_MODE_ENV}={HIP_BUILD_REQUIRED}; \
             set HIPCC=/path/to/hipcc or select {HIP_BUILD_CPU_ONLY}"
        ));
    }

    let hip_sources = walk_with_ext(&PathBuf::from("src"), "hip");
    let cpp_sources = walk_with_ext(&PathBuf::from("src"), "cpp");

    if hip_sources.is_empty() && cpp_sources.is_empty() {
        return Err(format!(
            "no HIP/CPP sources under src/ while {HIP_BUILD_MODE_ENV}={HIP_BUILD_REQUIRED}; \
             only {HIP_BUILD_CPU_ONLY} may produce no kernel archive"
        ));
    }

    write_row_format_header(&out_dir)?;
    compile_sources(&hipcc, &out_dir, &hip_sources, &cpp_sources)?;

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=logismos_kernels");
    // The HIP device runtime lives in the `amdhip64` shared object
    // already pulled in by hipcore; we rely on that.
    // hipcc links stdc++; re-export for the final binary.
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

fn write_row_format_header(out_dir: &Path) -> Result<(), String> {
    let derived = validate_row_format_layouts()?;
    let header = render_row_format_header(&derived);
    std::fs::write(out_dir.join("row_format.h"), header)
        .map_err(|error| format!("write generated serialized-row format header: {error}"))
}

struct RowFormatDerived {
    k_pair: KPairGeometry,
    q6_half: Q6HalfLayout,
}

struct KPairGeometry {
    count: usize,
    q4_values: usize,
    q4_quant_bytes: usize,
    q4_scale_bytes: usize,
}

struct Q6HalfLayout {
    count: usize,
    low_bytes: usize,
    high_bytes: usize,
    scales: usize,
    values: usize,
    values_per_scale: usize,
}

fn validate_row_format_layouts() -> Result<RowFormatDerived, String> {
    validate_q8_0_layout()?;
    let k_pair = validate_q4_k_layout()?;
    validate_q5_k_layout(&k_pair)?;
    let q6_half = validate_q6_k_layout()?;
    validate_iq4_nl_layout()?;
    validate_iq4_xs_layout()?;
    validate_fixed_width_fields()?;
    validate_serialized_row_sizes()?;

    Ok(RowFormatDerived { k_pair, q6_half })
}

fn validate_q8_0_layout() -> Result<(), String> {
    let values_per_block = quant::q8_0::Q8_0_VALUES_PER_BLOCK;
    let scale_bytes = quant::q8_0::Q8_0_SCALE_BYTES;
    let value_bytes = quant::q8_0::Q8_0_VALUE_BYTES;
    let block_bytes = quant::q8_0::Q8_0_BLOCK_BYTES;
    let derived_block_bytes = scale_bytes.checked_add(value_bytes).ok_or_else(|| {
        "quant Q8_0 header fields overflow while deriving block bytes".to_string()
    })?;
    if block_bytes != derived_block_bytes
        || value_bytes != values_per_block
        || scale_bytes != std::mem::size_of::<u16>()
    {
        return Err("quant Q8_0 constants violate their declared layout relation".to_string());
    }
    Ok(())
}

fn validate_q4_k_layout() -> Result<KPairGeometry, String> {
    let q4_values = quant::Q4_K_VALUES_PER_BLOCK;
    let q4_prefix = quant::q4_k::Q4_K_PREFIX_BYTES;
    let q4_scales = quant::q4_k::Q4_K_SCALE_BYTES;
    let q4_quant = quant::q4_k::Q4_K_QUANT_BYTES;
    let pair_width = checked_mul(quant::K_GROUP_VALUES, 2, "Q4_K pair width")?;
    let count = checked_div_exact(q4_values, pair_width, "Q4_K values per pair")?;
    let u8_bits = usize::try_from(u8::BITS)
        .map_err(|_| "u8 bit width does not fit usize while validating row formats".to_string())?;
    require_equal(
        checked_mul(count, 2, "Q4_K scale-pair bit capacity")?,
        u8_bits,
        "Q4_K scale pair count must fill the Q5_K fifth-bit byte",
    )?;
    require_equal(
        q4_quant,
        checked_mul(count, quant::K_GROUP_VALUES, "Q4_K quant bytes")?,
        "Q4_K quant bytes must encode one byte per pair lane",
    )?;
    require_equal(
        q4_scales,
        checked_mul(count, 3, "Q4_K scale bytes")?,
        "Q4_K scale bytes must encode three bytes per pair",
    )?;
    require_equal(
        quant::q4_k::Q4_K_SUPER_MINIMUM_OFFSET,
        std::mem::size_of::<u16>(),
        "Q4_K super-minimum offset must follow the super-scale",
    )?;
    require_equal(
        quant::q4_k::Q4_K_SCALE_OFFSET,
        q4_prefix,
        "Q4_K scale offset must follow the prefix",
    )?;
    require_equal(
        quant::q4_k::Q4_K_QUANT_OFFSET,
        checked_sum(
            &[quant::q4_k::Q4_K_SCALE_OFFSET, q4_scales],
            "Q4_K quant offset",
        )?,
        "Q4_K quant offset must follow scales",
    )?;

    Ok(KPairGeometry {
        count,
        q4_values,
        q4_quant_bytes: q4_quant,
        q4_scale_bytes: q4_scales,
    })
}

fn validate_q5_k_layout(k_pair: &KPairGeometry) -> Result<(), String> {
    let q5_prefix = quant::q5_k::Q5_K_PREFIX_BYTES;
    let q5_scales = quant::q5_k::Q5_K_SCALE_BYTES;
    let q5_high = quant::q5_k::Q5_K_HIGH_BITS_BYTES;
    require_equal(
        quant::Q5_K_VALUES_PER_BLOCK,
        k_pair.q4_values,
        "Q5_K values must share Q4_K pair geometry",
    )?;
    require_equal(
        quant::q5_k::Q5_K_QUANT_BYTES,
        k_pair.q4_quant_bytes,
        "Q5_K quant bytes must share Q4_K pair geometry",
    )?;
    require_equal(
        q5_scales,
        k_pair.q4_scale_bytes,
        "Q5_K scale bytes must share Q4_K pair geometry",
    )?;
    require_equal(
        q5_high,
        quant::K_GROUP_VALUES,
        "Q5_K fifth-bit bytes must encode one byte per group lane",
    )?;
    require_equal(
        quant::q5_k::Q5_K_SUPER_MINIMUM_OFFSET,
        std::mem::size_of::<u16>(),
        "Q5_K super-minimum offset must follow the super-scale",
    )?;
    require_equal(
        quant::q5_k::Q5_K_SCALE_OFFSET,
        q5_prefix,
        "Q5_K scale offset must follow the prefix",
    )?;
    require_equal(
        quant::q5_k::Q5_K_HIGH_BITS_OFFSET,
        checked_sum(
            &[quant::q5_k::Q5_K_SCALE_OFFSET, q5_scales],
            "Q5_K high-bit offset",
        )?,
        "Q5_K high-bit offset must follow scales",
    )?;
    require_equal(
        quant::q5_k::Q5_K_QUANT_OFFSET,
        checked_sum(
            &[quant::q5_k::Q5_K_HIGH_BITS_OFFSET, q5_high],
            "Q5_K quant offset",
        )?,
        "Q5_K quant offset must follow fifth-bit planes",
    )
}

fn validate_q6_k_layout() -> Result<Q6HalfLayout, String> {
    let values = quant::Q6_K_VALUES_PER_BLOCK;
    let values_per_quarter = quant::q6_k::Q6_K_VALUES_PER_QUARTER;
    let quarters_per_half = quant::q6_k::Q6_K_QUARTERS_PER_HALF_BLOCK;
    let low = quant::q6_k::Q6_K_LOW_BITS_BYTES;
    let high = quant::q6_k::Q6_K_HIGH_BITS_BYTES;
    let scales = quant::q6_k::Q6_K_SCALE_BYTES;
    let half_width = checked_mul(
        values_per_quarter,
        quarters_per_half,
        "Q6_K values per half block",
    )?;
    let u8_bits = usize::try_from(u8::BITS)
        .map_err(|_| "u8 bit width does not fit usize while validating row formats".to_string())?;
    require_equal(
        checked_mul(quarters_per_half, 2, "Q6_K high-plane bit capacity")?,
        u8_bits,
        "Q6_K quarter count must fill its two-bit high-plane byte",
    )?;
    let count = checked_div_exact(values, half_width, "Q6_K half blocks")?;
    let low_bytes = checked_div_exact(low, count, "Q6_K low bytes per half")?;
    let high_bytes = checked_div_exact(high, count, "Q6_K high bytes per half")?;
    let scales_per_half = checked_div_exact(scales, count, "Q6_K scales per half")?;
    let values_per_half = checked_div_exact(values, count, "Q6_K values per half")?;
    let values_per_scale = checked_div_exact(values, scales, "Q6_K values per scale")?;
    require_equal(
        low_bytes,
        checked_mul(values_per_quarter, 2, "Q6_K low bytes per half relation")?,
        "Q6_K low planes must encode two bytes per quarter lane",
    )?;
    require_equal(
        high_bytes,
        values_per_quarter,
        "Q6_K high planes must encode one byte per quarter lane",
    )?;
    require_equal(
        scales_per_half,
        checked_mul(quarters_per_half, 2, "Q6_K scales per half relation")?,
        "Q6_K scales must encode two values per quarter",
    )?;
    require_equal(
        values_per_quarter,
        checked_mul(values_per_scale, 2, "Q6_K values per quarter relation")?,
        "Q6_K quarters must contain two scale groups",
    )?;
    require_equal(
        quant::q6_k::Q6_K_HIGH_BITS_OFFSET,
        low,
        "Q6_K high-bit offset must follow low planes",
    )?;
    require_equal(
        quant::q6_k::Q6_K_SCALE_OFFSET,
        checked_sum(
            &[quant::q6_k::Q6_K_HIGH_BITS_OFFSET, high],
            "Q6_K scale offset",
        )?,
        "Q6_K scale offset must follow high planes",
    )?;
    require_equal(
        quant::q6_k::Q6_K_SUPER_SCALE_OFFSET,
        checked_sum(
            &[quant::q6_k::Q6_K_SCALE_OFFSET, scales],
            "Q6_K super-scale offset",
        )?,
        "Q6_K super-scale offset must follow signed scales",
    )?;

    Ok(Q6HalfLayout {
        count,
        low_bytes,
        high_bytes,
        scales: scales_per_half,
        values: values_per_half,
        values_per_scale,
    })
}

fn validate_iq4_nl_layout() -> Result<(), String> {
    let values = quant::IQ4_NL_VALUES_PER_BLOCK;
    let quant_bytes = quant::iq4_nl::IQ4_NL_QUANT_BYTES;
    require_equal(
        checked_mul(quant_bytes, 2, "IQ4_NL reconstructed values")?,
        values,
        "IQ4_NL quant bytes must encode two reconstruction indices",
    )?;
    require_equal(
        quant::iq4_nl::IQ4_NL_QUANT_OFFSET,
        quant::iq4_nl::IQ4_NL_SCALE_BYTES,
        "IQ4_NL quant offset must follow the block scale",
    )
}

fn validate_iq4_xs_layout() -> Result<(), String> {
    let values = quant::IQ4_XS_VALUES_PER_BLOCK;
    let group_values = quant::iq4_xs::IQ4_XS_GROUP_VALUES;
    let scale_low = quant::iq4_xs::IQ4_XS_SCALE_LOW_BYTES;
    let scale_high = quant::iq4_xs::IQ4_XS_SCALE_HIGH_BYTES;
    let group_count = checked_div_exact(values, group_values, "IQ4_XS group count")?;
    if !group_values.is_multiple_of(2) {
        return Err(
            "quant layout relation violated: IQ4_XS group width must be even for packed lanes"
                .to_string(),
        );
    }
    require_equal(
        checked_mul(
            quant::iq4_xs::IQ4_XS_QUANT_BYTES,
            2,
            "IQ4_XS reconstructed values",
        )?,
        values,
        "IQ4_XS quant bytes must encode two reconstruction indices",
    )?;
    require_equal(
        checked_mul(scale_low, 2, "IQ4_XS low-scale groups")?,
        group_count,
        "IQ4_XS low scale bytes must encode two groups",
    )?;
    require_equal(
        checked_mul(scale_high, 4, "IQ4_XS high-scale groups")?,
        group_count,
        "IQ4_XS high scale bytes must encode four groups",
    )?;
    require_equal(
        quant::iq4_xs::IQ4_XS_SCALE_HIGH_OFFSET,
        quant::iq4_xs::IQ4_XS_SCALE_BYTES,
        "IQ4_XS high-scale offset must follow the block scale",
    )?;
    require_equal(
        quant::iq4_xs::IQ4_XS_SCALE_LOW_OFFSET,
        checked_sum(
            &[quant::iq4_xs::IQ4_XS_SCALE_HIGH_OFFSET, scale_high],
            "IQ4_XS low-scale offset",
        )?,
        "IQ4_XS low-scale offset must follow high scale bits",
    )?;
    require_equal(
        quant::iq4_xs::IQ4_XS_QUANT_OFFSET,
        checked_sum(
            &[quant::iq4_xs::IQ4_XS_SCALE_LOW_OFFSET, scale_low],
            "IQ4_XS quant offset",
        )?,
        "IQ4_XS quant offset must follow low scale bits",
    )?;
    Ok(())
}

fn validate_fixed_width_fields() -> Result<(), String> {
    if quant::f32_row::F32_ROW_VALUE_BYTES != std::mem::size_of::<u32>()
        || quant::q4_k::Q4_K_PREFIX_BYTES != std::mem::size_of::<u16>() * 2
        || quant::q5_k::Q5_K_PREFIX_BYTES != std::mem::size_of::<u16>() * 2
        || quant::q6_k::Q6_K_SUPER_SCALE_BYTES != std::mem::size_of::<u16>()
        || quant::iq4_nl::IQ4_NL_SCALE_BYTES != std::mem::size_of::<u16>()
        || quant::iq4_xs::IQ4_XS_SCALE_BYTES != std::mem::size_of::<u16>()
        || quant::iq4_xs::IQ4_XS_SCALE_HIGH_BYTES != std::mem::size_of::<u16>()
    {
        return Err(
            "quant constants violate serialized-row fixed-width field representation".to_string(),
        );
    }
    Ok(())
}

fn validate_serialized_row_sizes() -> Result<(), String> {
    let q4_derived = checked_sum(
        &[
            quant::q4_k::Q4_K_PREFIX_BYTES,
            quant::q4_k::Q4_K_SCALE_BYTES,
            quant::q4_k::Q4_K_QUANT_BYTES,
        ],
        "Q4_K",
    )?;
    let q5_derived = checked_sum(
        &[
            quant::q5_k::Q5_K_PREFIX_BYTES,
            quant::q5_k::Q5_K_SCALE_BYTES,
            quant::q5_k::Q5_K_HIGH_BITS_BYTES,
            quant::q5_k::Q5_K_QUANT_BYTES,
        ],
        "Q5_K",
    )?;
    let q6_derived = checked_sum(
        &[
            quant::q6_k::Q6_K_LOW_BITS_BYTES,
            quant::q6_k::Q6_K_HIGH_BITS_BYTES,
            quant::q6_k::Q6_K_SCALE_BYTES,
            quant::q6_k::Q6_K_SUPER_SCALE_BYTES,
        ],
        "Q6_K",
    )?;
    let iq4_nl_derived = checked_sum(
        &[
            quant::iq4_nl::IQ4_NL_SCALE_BYTES,
            quant::iq4_nl::IQ4_NL_QUANT_BYTES,
        ],
        "IQ4_NL",
    )?;
    let iq4_xs_derived = checked_sum(
        &[
            quant::iq4_xs::IQ4_XS_SCALE_BYTES,
            quant::iq4_xs::IQ4_XS_SCALE_LOW_BYTES,
            quant::iq4_xs::IQ4_XS_SCALE_HIGH_BYTES,
            quant::iq4_xs::IQ4_XS_QUANT_BYTES,
        ],
        "IQ4_XS",
    )?;
    if quant::Q4_K_BLOCK_BYTES != q4_derived
        || quant::Q5_K_BLOCK_BYTES != q5_derived
        || quant::Q6_K_BLOCK_BYTES != q6_derived
        || quant::IQ4_NL_BLOCK_BYTES != iq4_nl_derived
        || quant::IQ4_XS_BLOCK_BYTES != iq4_xs_derived
    {
        return Err(
            "quant constants violate an executable serialized-row layout relation".to_string(),
        );
    }
    Ok(())
}

fn render_row_format_header(derived: &RowFormatDerived) -> String {
    format!(
        "{}{}{}{}{}",
        render_header_prefix(),
        render_k_format_constants(),
        render_q6_format_constants(),
        render_iq4_format_constants(),
        render_derived_constants(derived),
    )
}

fn render_header_prefix() -> String {
    format!(
        "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\ninline constexpr std::size_t LOGISMOS_F32_VALUE_BYTES = {f32_bytes};\ninline constexpr std::size_t LOGISMOS_Q8_0_VALUES_PER_BLOCK = {values_per_block};\ninline constexpr std::size_t LOGISMOS_Q8_0_SCALE_BYTES = {scale_bytes};\ninline constexpr std::size_t LOGISMOS_Q8_0_VALUE_BYTES = {value_bytes};\ninline constexpr std::size_t LOGISMOS_Q8_0_BLOCK_BYTES = {block_bytes};\ninline constexpr std::size_t LOGISMOS_K_GROUP_VALUES = {k_group};\n",
        f32_bytes = quant::f32_row::F32_ROW_VALUE_BYTES,
        values_per_block = quant::q8_0::Q8_0_VALUES_PER_BLOCK,
        scale_bytes = quant::q8_0::Q8_0_SCALE_BYTES,
        value_bytes = quant::q8_0::Q8_0_VALUE_BYTES,
        block_bytes = quant::q8_0::Q8_0_BLOCK_BYTES,
        k_group = quant::K_GROUP_VALUES,
    )
}

fn render_k_format_constants() -> String {
    format!(
        "inline constexpr std::size_t LOGISMOS_Q4_K_VALUES_PER_BLOCK = {q4_values};\ninline constexpr std::size_t LOGISMOS_Q4_K_PREFIX_BYTES = {q4_prefix};\ninline constexpr std::size_t LOGISMOS_Q4_K_SCALE_BYTES = {q4_scales};\ninline constexpr std::size_t LOGISMOS_Q4_K_BLOCK_BYTES = {q4_bytes};\ninline constexpr std::size_t LOGISMOS_Q5_K_VALUES_PER_BLOCK = {q5_values};\ninline constexpr std::size_t LOGISMOS_Q5_K_PREFIX_BYTES = {q5_prefix};\ninline constexpr std::size_t LOGISMOS_Q5_K_SCALE_BYTES = {q5_scales};\ninline constexpr std::size_t LOGISMOS_Q5_K_HIGH_BITS_BYTES = {q5_high};\ninline constexpr std::size_t LOGISMOS_Q5_K_BLOCK_BYTES = {q5_bytes};\n",
        q4_values = quant::Q4_K_VALUES_PER_BLOCK,
        q4_prefix = quant::q4_k::Q4_K_PREFIX_BYTES,
        q4_scales = quant::q4_k::Q4_K_SCALE_BYTES,
        q4_bytes = quant::Q4_K_BLOCK_BYTES,
        q5_values = quant::Q5_K_VALUES_PER_BLOCK,
        q5_prefix = quant::q5_k::Q5_K_PREFIX_BYTES,
        q5_scales = quant::q5_k::Q5_K_SCALE_BYTES,
        q5_high = quant::q5_k::Q5_K_HIGH_BITS_BYTES,
        q5_bytes = quant::Q5_K_BLOCK_BYTES,
    )
}

fn render_q6_format_constants() -> String {
    format!(
        "inline constexpr std::size_t LOGISMOS_Q6_K_VALUES_PER_BLOCK = {values};\ninline constexpr std::size_t LOGISMOS_Q6_K_LOW_BITS_BYTES = {low};\ninline constexpr std::size_t LOGISMOS_Q6_K_HIGH_BITS_BYTES = {high};\ninline constexpr std::size_t LOGISMOS_Q6_K_SCALE_BYTES = {scales};\ninline constexpr std::size_t LOGISMOS_Q6_K_BLOCK_BYTES = {bytes};\n",
        values = quant::Q6_K_VALUES_PER_BLOCK,
        low = quant::q6_k::Q6_K_LOW_BITS_BYTES,
        high = quant::q6_k::Q6_K_HIGH_BITS_BYTES,
        scales = quant::q6_k::Q6_K_SCALE_BYTES,
        bytes = quant::Q6_K_BLOCK_BYTES,
    )
}

fn render_iq4_format_constants() -> String {
    let reconstruction_values = quant::IQ4_RECONSTRUCTION_VALUES
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "inline constexpr std::size_t LOGISMOS_IQ4_NL_VALUES_PER_BLOCK = {nl_values};\ninline constexpr std::size_t LOGISMOS_IQ4_NL_BLOCK_BYTES = {nl_bytes};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_VALUES_PER_BLOCK = {xs_values};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_BLOCK_BYTES = {xs_bytes};\ninline constexpr std::int8_t LOGISMOS_IQ4_RECONSTRUCTION_VALUES[16] = {{{reconstruction_values}}};\n",
        nl_values = quant::IQ4_NL_VALUES_PER_BLOCK,
        nl_bytes = quant::IQ4_NL_BLOCK_BYTES,
        xs_values = quant::IQ4_XS_VALUES_PER_BLOCK,
        xs_bytes = quant::IQ4_XS_BLOCK_BYTES,
    )
}

fn render_derived_constants(derived: &RowFormatDerived) -> String {
    format!(
        "inline constexpr std::size_t LOGISMOS_K_PAIR_COUNT = {k_pair_count};\ninline constexpr std::size_t LOGISMOS_Q4_K_SUPER_MINIMUM_OFFSET = {q4_super_minimum_offset};\ninline constexpr std::size_t LOGISMOS_Q4_K_SCALE_OFFSET = {q4_scale_offset};\ninline constexpr std::size_t LOGISMOS_Q4_K_QUANT_OFFSET = {q4_quant_offset};\ninline constexpr std::size_t LOGISMOS_Q5_K_SUPER_MINIMUM_OFFSET = {q5_super_minimum_offset};\ninline constexpr std::size_t LOGISMOS_Q5_K_SCALE_OFFSET = {q5_scale_offset};\ninline constexpr std::size_t LOGISMOS_Q5_K_HIGH_BITS_OFFSET = {q5_high_offset};\ninline constexpr std::size_t LOGISMOS_Q5_K_QUANT_OFFSET = {q5_quant_offset};\ninline constexpr std::size_t LOGISMOS_Q6_K_HIGH_BITS_OFFSET = {q6_high_offset};\ninline constexpr std::size_t LOGISMOS_Q6_K_SCALE_OFFSET = {q6_scale_offset};\ninline constexpr std::size_t LOGISMOS_Q6_K_SUPER_SCALE_OFFSET = {q6_super_scale_offset};\ninline constexpr std::size_t LOGISMOS_Q6_K_VALUES_PER_QUARTER = {q6_values_per_quarter};\ninline constexpr std::size_t LOGISMOS_Q6_K_QUARTERS_PER_HALF_BLOCK = {q6_quarters_per_half};\ninline constexpr std::size_t LOGISMOS_Q6_K_HALF_BLOCK_COUNT = {q6_half_block_count};\ninline constexpr std::size_t LOGISMOS_Q6_K_LOW_BYTES_PER_HALF = {q6_low_bytes_per_half};\ninline constexpr std::size_t LOGISMOS_Q6_K_HIGH_BYTES_PER_HALF = {q6_high_bytes_per_half};\ninline constexpr std::size_t LOGISMOS_Q6_K_SCALES_PER_HALF = {q6_scales_per_half};\ninline constexpr std::size_t LOGISMOS_Q6_K_VALUES_PER_HALF = {q6_values_per_half};\ninline constexpr std::size_t LOGISMOS_Q6_K_VALUES_PER_SCALE = {q6_values_per_scale};\ninline constexpr std::size_t LOGISMOS_IQ4_NL_QUANT_OFFSET = {iq4_nl_quant_offset};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_SCALE_HIGH_OFFSET = {iq4_xs_scale_high_offset};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_SCALE_LOW_OFFSET = {iq4_xs_scale_low_offset};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_QUANT_OFFSET = {iq4_xs_quant_offset};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_GROUP_VALUES = {iq4_xs_group_values};\n",
        k_pair_count = derived.k_pair.count,
        q4_super_minimum_offset = quant::q4_k::Q4_K_SUPER_MINIMUM_OFFSET,
        q4_scale_offset = quant::q4_k::Q4_K_SCALE_OFFSET,
        q4_quant_offset = quant::q4_k::Q4_K_QUANT_OFFSET,
        q5_super_minimum_offset = quant::q5_k::Q5_K_SUPER_MINIMUM_OFFSET,
        q5_scale_offset = quant::q5_k::Q5_K_SCALE_OFFSET,
        q5_high_offset = quant::q5_k::Q5_K_HIGH_BITS_OFFSET,
        q5_quant_offset = quant::q5_k::Q5_K_QUANT_OFFSET,
        q6_high_offset = quant::q6_k::Q6_K_HIGH_BITS_OFFSET,
        q6_scale_offset = quant::q6_k::Q6_K_SCALE_OFFSET,
        q6_super_scale_offset = quant::q6_k::Q6_K_SUPER_SCALE_OFFSET,
        q6_values_per_quarter = quant::q6_k::Q6_K_VALUES_PER_QUARTER,
        q6_quarters_per_half = quant::q6_k::Q6_K_QUARTERS_PER_HALF_BLOCK,
        q6_half_block_count = derived.q6_half.count,
        q6_low_bytes_per_half = derived.q6_half.low_bytes,
        q6_high_bytes_per_half = derived.q6_half.high_bytes,
        q6_scales_per_half = derived.q6_half.scales,
        q6_values_per_half = derived.q6_half.values,
        q6_values_per_scale = derived.q6_half.values_per_scale,
        iq4_nl_quant_offset = quant::iq4_nl::IQ4_NL_QUANT_OFFSET,
        iq4_xs_scale_high_offset = quant::iq4_xs::IQ4_XS_SCALE_HIGH_OFFSET,
        iq4_xs_scale_low_offset = quant::iq4_xs::IQ4_XS_SCALE_LOW_OFFSET,
        iq4_xs_quant_offset = quant::iq4_xs::IQ4_XS_QUANT_OFFSET,
        iq4_xs_group_values = quant::iq4_xs::IQ4_XS_GROUP_VALUES,
    )
}

fn checked_sum(fields: &[usize], format: &str) -> Result<usize, String> {
    fields.iter().try_fold(0usize, |total, field| {
        total.checked_add(*field).ok_or_else(|| {
            format!("quant {format} header fields overflow while deriving block bytes")
        })
    })
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("quant layout relation overflow while deriving {label}"))
}

fn checked_div_exact(dividend: usize, divisor: usize, label: &str) -> Result<usize, String> {
    if divisor == 0 {
        return Err(format!(
            "quant layout relation has zero divisor while deriving {label}"
        ));
    }
    if !dividend.is_multiple_of(divisor) {
        return Err(format!(
            "quant layout relation is not exact while deriving {label}"
        ));
    }
    Ok(dividend / divisor)
}

fn require_equal(actual: usize, expected: usize, relation: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "quant layout relation violated: {relation} ({actual} != {expected})"
        ));
    }
    Ok(())
}

fn compile_sources(
    hipcc: &str,
    out_dir: &Path,
    hip_sources: &[PathBuf],
    cpp_sources: &[PathBuf],
) -> Result<(), String> {
    let mut obj_files = Vec::with_capacity(hip_sources.len() + cpp_sources.len());

    for src in hip_sources.iter().chain(cpp_sources) {
        let obj = out_dir.join(format!(
            "{}.o",
            src.file_name().and_then(|s| s.to_str()).unwrap_or("anon")
        ));
        // kanon:ignore RUST/no-direct-process-command -- invoking hipcc is the build script's purpose
        let mut command = Command::new(hipcc);
        command
            .args([
                &format!("--offload-arch={HIP_TARGET}"),
                "-O3",
                "-std=c++17",
                "-fPIC",
                // Pin wave32 on gfx11 for WMMA correctness (dossier 01
                // §3.3 + §7.4). RDNA3 defaults to wave32 anyway; this
                // makes the choice audit-visible.
                "-mno-wavefrontsize64",
                "-I",
            ])
            .arg(out_dir);
        if src.file_name().is_some_and(|name| {
            name == "row_gemv.hip" || name == "gdn_step.hip" || name == "causal_conv_step.hip"
        }) {
            // WHY: these correctness baselines retain separately rounded f32
            // operations. Scope no-fast-math and no contraction to their
            // sources rather than changing the rest of the HIP archive.
            command.args(["-fno-fast-math", "-ffp-contract=off"]);
        }
        // kanon:ignore RUST/no-direct-process-command -- invoking hipcc is the build script's purpose
        let status = match command.arg("-c").arg(src).arg("-o").arg(&obj).status() {
            Ok(s) => s,
            Err(e) => return Err(format!("invoke hipcc on {}: {e}", src.display())),
        };
        if !status.success() {
            return Err(format!(
                "hipcc failed compiling {} (status {status:?})",
                src.display()
            ));
        }
        obj_files.push(obj);
    }

    let archive = out_dir.join("liblogismos_kernels.a");
    remove_stale_archive(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = match Command::new(&ar)
        .arg("rcs")
        .arg(&archive)
        .args(&obj_files)
        .status()
    {
        Ok(s) => s,
        Err(e) => return Err(format!("invoke ar at {ar}: {e}")),
    };
    if !status.success() {
        return Err(format!("ar failed (status {status:?})"));
    }

    Ok(())
}

fn gpu_feature_enabled() -> bool {
    env::var_os("CARGO_FEATURE_GPU").is_some()
}

fn hip_build_mode() -> Result<HipBuildConfiguration, String> {
    match env::var(HIP_BUILD_MODE_ENV) {
        Ok(value) if value == HIP_BUILD_REQUIRED => Ok(HipBuildConfiguration {
            mode: HipBuildMode::Required,
            explicitly_set: true,
        }),
        Ok(value) if value == HIP_BUILD_CPU_ONLY => Ok(HipBuildConfiguration {
            mode: HipBuildMode::CpuOnly,
            explicitly_set: true,
        }),
        Ok(value) => Err(format!(
            "invalid {HIP_BUILD_MODE_ENV}={value:?}; expected {HIP_BUILD_REQUIRED} or {HIP_BUILD_CPU_ONLY}"
        )),
        Err(env::VarError::NotPresent) => Ok(HipBuildConfiguration {
            mode: HipBuildMode::Required,
            explicitly_set: false,
        }),
        Err(error) => Err(format!("read {HIP_BUILD_MODE_ENV}: {error}")),
    }
}

/// Remove a stale static archive, logging (but not failing) on I/O error.
///
/// WHY: the archive is routinely replaced on every build; failure to remove
/// a prior copy is non-fatal because `ar rcs` overwrites. Surfacing the error
/// as `cargo:warning=...` lets an operator diagnose downstream `ar` failures
/// without silently discarding the root cause.
fn remove_stale_archive(archive: &Path) {
    if !archive.exists() {
        return;
    }
    if let Err(err) = std::fs::remove_file(archive) {
        println!(
            "cargo:warning=failed to remove stale archive {}: {err}",
            archive.display()
        );
    }
}

fn which(cmd: &str) -> Option<PathBuf> {
    if cmd.contains('/') {
        let p = PathBuf::from(cmd);
        return if p.is_file() { Some(p) } else { None };
    }
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn walk_sources(root: &Path) -> Vec<PathBuf> {
    let mut hip = walk_with_ext(root, "hip");
    hip.extend(walk_with_ext(root, "cpp"));
    hip.extend(walk_with_ext(root, "h"));
    hip.extend(walk_with_ext(root, "hpp"));
    hip
}

fn walk_with_ext(root: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_rec(root, ext, &mut out);
    out
}

fn walk_rec(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk_rec(&p, ext, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(p);
        }
    }
}
