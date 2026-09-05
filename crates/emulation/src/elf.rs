//! Bounded inspection of CPU-assembled AMDGPU relocatable text fixtures.
//!
//! This module deliberately accepts `ET_REL` objects only. It is not an HSA
//! code-object loader: it rejects program headers, relocations, descriptors, and
//! every section other than the inert `.strtab`/`.symtab` pair and one `.text`.

use crate::{
    ElfTooLargeSnafu, ElfTruncatedSnafu, MissingExecutableTextSnafu,
    ProgramHeadersUnsupportedSnafu, RelocationsUnsupportedSnafu, Result, UnsupportedElfHeaderSnafu,
    UnsupportedElfIdentSnafu, UnsupportedElfSectionSnafu, UnsupportedTextSectionSnafu,
    Wave32Program,
};

/// Largest ELF object this inspect-only boundary accepts.
pub const MAX_ELF_OBJECT_BYTES: usize = 16 * 1024;

const ELF_HEADER_BYTES: usize = 64;
const SECTION_HEADER_BYTES: usize = 64;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ELFOSABI_AMDGPU_HSA: u8 = 0x40;
const AMDGPU_HSA_ABI_VERSION: u8 = 4;
const ET_REL: u16 = 1;
const EM_AMDGPU: u16 = 224;
const GFX1100_ELF_FLAGS: u32 = 0x41;
const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_REL: u32 = 9;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

/// Inspectable `.text` bytes from a strict AMDGPU relocatable fixture.
///
/// Constructed only by [`inspect_relocatable_text`]. It is still not a runnable
/// HSA kernel: dispatch is CPU-only and uses the raw-instruction executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelocatableText {
    text: Vec<u8>,
}

impl RelocatableText {
    /// Returns the admitted raw `.text` instruction bytes.
    #[must_use]
    pub fn text(&self) -> &[u8] {
        &self.text
    }

    /// Converts the admitted text into the CPU-only wave32 dispatch contract.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if the supplied register state or step bound is
    /// invalid for the raw-instruction executor.
    pub fn into_wave32_program(
        self,
        initial_registers: Vec<[u32; crate::WAVE32_LANES]>,
        max_steps: usize,
    ) -> Result<Wave32Program> {
        Wave32Program::new(self.text, initial_registers, max_steps)
    }
}

/// Inspects one strict, non-runnable, gfx1100 relocatable text fixture.
///
/// The admission contract is an ELF64 little-endian `ET_REL` AMDGPU-HSA object
/// with the LLVM-observed gfx1100 flags, no program headers or relocations, and
/// only `.strtab`, `.symtab`, and one executable four-byte-aligned `.text`.
/// It is intentionally narrower than AMDHSA code objects and must not be used to
/// claim support for kernel descriptors, metadata, memory, or GPU dispatch.
///
/// # Errors
///
/// Returns [`crate::Error`] for malformed/truncated ELF, another target or ABI,
/// program headers, relocations, unsupported sections, or invalid `.text`.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded ELF admission sequence keeps every header and section rejection visible in one audit path"
)]
pub fn inspect_relocatable_text(object: &[u8]) -> Result<RelocatableText> {
    if object.len() > MAX_ELF_OBJECT_BYTES {
        return Err(ElfTooLargeSnafu {
            actual: object.len(),
            maximum: MAX_ELF_OBJECT_BYTES,
        }
        .build());
    }
    let header = bytes_at(object, 0, ELF_HEADER_BYTES, "ELF header")?;
    check_ident(header)?;
    check_header(header)?;

    let section_offset = usize_from(
        read_u64(header, 40, "section-header offset")?,
        "section-header offset",
    )?;
    let section_entry_size = usize::from(read_u16(header, 58, "section-header entry size")?);
    let section_count = usize::from(read_u16(header, 60, "section count")?);
    let section_names = usize::from(read_u16(header, 62, "section-name table index")?);
    if section_entry_size != SECTION_HEADER_BYTES {
        return Err(UnsupportedElfHeaderSnafu {
            field: "section-header entry size",
            actual: u64::from(read_u16(header, 58, "section-header entry size")?),
            expected: u64::try_from(SECTION_HEADER_BYTES).unwrap_or(u64::MAX),
        }
        .build());
    }
    if section_count == 0 || section_count > 16 || section_names >= section_count {
        return Err(UnsupportedElfHeaderSnafu {
            field: "section table shape",
            actual: u64::try_from(section_count).unwrap_or(u64::MAX),
            expected: 1u64,
        }
        .build());
    }
    let section_table_bytes = section_count
        .checked_mul(SECTION_HEADER_BYTES)
        .ok_or_else(|| {
            ElfTruncatedSnafu {
                context: "section-header table size",
                offset: section_offset,
            }
            .build()
        })?;
    bytes_at(
        object,
        section_offset,
        section_table_bytes,
        "section-header table",
    )?;

    let mut sections = Vec::with_capacity(section_count);
    for index in 0..section_count {
        let offset = section_offset
            .checked_add(index.checked_mul(SECTION_HEADER_BYTES).ok_or_else(|| {
                ElfTruncatedSnafu {
                    context: "section-header offset",
                    offset: section_offset,
                }
                .build()
            })?)
            .ok_or_else(|| {
                ElfTruncatedSnafu {
                    context: "section-header offset",
                    offset: section_offset,
                }
                .build()
            })?;
        sections.push(Section::parse(bytes_at(
            object,
            offset,
            SECTION_HEADER_BYTES,
            "section header",
        )?)?);
    }

    let Some(names_section) = sections.get(section_names) else {
        return Err(ElfTruncatedSnafu {
            context: "section-name table",
            offset: section_names,
        }
        .build());
    };
    if names_section.kind != SHT_STRTAB {
        return Err(UnsupportedElfSectionSnafu {
            section: "section-name table".to_owned(),
            section_type: names_section.kind,
        }
        .build());
    }
    let names = bytes_at(
        object,
        names_section.offset,
        names_section.size,
        "section-name table",
    )?;
    if section_name(names, names_section.name_offset)? != b".strtab" || names_section.flags != 0 {
        return Err(UnsupportedElfSectionSnafu {
            section: printable_name(section_name(names, names_section.name_offset)?),
            section_type: names_section.kind,
        }
        .build());
    }

    let mut text = None;
    for (index, section) in sections.iter().enumerate() {
        let name = section_name(names, section.name_offset)?;
        if index == 0 {
            if section.kind != SHT_NULL {
                return Err(UnsupportedElfSectionSnafu {
                    section: printable_name(name),
                    section_type: section.kind,
                }
                .build());
            }
            continue;
        }
        if index == section_names {
            continue;
        }
        if section.kind == SHT_RELA || section.kind == SHT_REL {
            return Err(RelocationsUnsupportedSnafu {
                section: printable_name(name),
            }
            .build());
        }
        if name == b".symtab" && section.kind == SHT_SYMTAB && section.flags == 0 {
            continue;
        }
        if name != b".text" || section.kind != SHT_PROGBITS {
            return Err(UnsupportedElfSectionSnafu {
                section: printable_name(name),
                section_type: section.kind,
            }
            .build());
        }
        if text.is_some() || section.flags != SHF_ALLOC | SHF_EXECINSTR || section.alignment != 4 {
            return Err(UnsupportedTextSectionSnafu {
                reason: "duplicate text, flags, or alignment",
            }
            .build());
        }
        let bytes = bytes_at(object, section.offset, section.size, ".text")?;
        if bytes.is_empty() || bytes.len() > crate::MAX_TEXT_BYTES || !bytes.len().is_multiple_of(4)
        {
            return Err(UnsupportedTextSectionSnafu {
                reason: "text must be non-empty, bounded, and four-byte aligned",
            }
            .build());
        }
        text = Some(bytes.to_vec());
    }

    let Some(text) = text else {
        return Err(MissingExecutableTextSnafu.build());
    };
    Ok(RelocatableText { text })
}

#[derive(Debug)]
struct Section {
    name_offset: usize,
    kind: u32,
    flags: u64,
    offset: usize,
    size: usize,
    alignment: u64,
}

impl Section {
    fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            name_offset: usize_from(
                read_u32(bytes, 0, "section name offset")?.into(),
                "section name offset",
            )?,
            kind: read_u32(bytes, 4, "section type")?,
            flags: read_u64(bytes, 8, "section flags")?,
            offset: usize_from(read_u64(bytes, 24, "section offset")?, "section offset")?,
            size: usize_from(read_u64(bytes, 32, "section size")?, "section size")?,
            alignment: read_u64(bytes, 48, "section alignment")?,
        })
    }
}

fn check_ident(header: &[u8]) -> Result<()> {
    let expected = [0x7f, b'E', b'L', b'F'];
    for (index, byte) in expected.iter().enumerate() {
        check_ident_byte(header, index, *byte, "magic")?;
    }
    check_ident_byte(header, 4, ELFCLASS64, "class")?;
    check_ident_byte(header, 5, ELFDATA2LSB, "data")?;
    check_ident_byte(header, 6, EV_CURRENT, "version")?;
    check_ident_byte(header, 7, ELFOSABI_AMDGPU_HSA, "OSABI")?;
    check_ident_byte(header, 8, AMDGPU_HSA_ABI_VERSION, "ABI version")
}

fn check_ident_byte(header: &[u8], index: usize, expected: u8, field: &'static str) -> Result<()> {
    let Some(actual) = header.get(index) else {
        return Err(ElfTruncatedSnafu {
            context: "ELF identification",
            offset: index,
        }
        .build());
    };
    if *actual != expected {
        return Err(UnsupportedElfIdentSnafu {
            field,
            actual: *actual,
            expected,
        }
        .build());
    }
    Ok(())
}

fn check_header(header: &[u8]) -> Result<()> {
    check_header_field(
        u64::from(read_u16(header, 16, "ELF type")?),
        u64::from(ET_REL),
        "type",
    )?;
    check_header_field(
        u64::from(read_u16(header, 18, "ELF machine")?),
        u64::from(EM_AMDGPU),
        "machine",
    )?;
    check_header_field(
        u64::from(read_u32(header, 20, "ELF version")?),
        1,
        "version",
    )?;
    check_header_field(read_u64(header, 24, "entry address")?, 0, "entry address")?;
    check_header_field(
        read_u64(header, 32, "program-header offset")?,
        0,
        "program-header offset",
    )?;
    let program_count = read_u16(header, 56, "program-header count")?;
    if program_count != 0 {
        return Err(ProgramHeadersUnsupportedSnafu {
            count: program_count,
        }
        .build());
    }
    check_header_field(
        u64::from(read_u32(header, 48, "target flags")?),
        u64::from(GFX1100_ELF_FLAGS),
        "gfx1100 target flags",
    )?;
    check_header_field(
        u64::from(read_u16(header, 52, "ELF header size")?),
        u64::try_from(ELF_HEADER_BYTES).unwrap_or(u64::MAX),
        "ELF header size",
    )
}

fn check_header_field(actual: u64, expected: u64, field: &'static str) -> Result<()> {
    if actual != expected {
        return Err(UnsupportedElfHeaderSnafu {
            field,
            actual,
            expected,
        }
        .build());
    }
    Ok(())
}

fn section_name(names: &[u8], offset: usize) -> Result<&[u8]> {
    let Some(bytes) = names.get(offset..) else {
        return Err(ElfTruncatedSnafu {
            context: "section name offset",
            offset,
        }
        .build());
    };
    let Some(terminator) = bytes.iter().position(|byte| *byte == 0) else {
        return Err(ElfTruncatedSnafu {
            context: "section name terminator",
            offset,
        }
        .build());
    };
    Ok(&bytes[..terminator])
}

fn printable_name(name: &[u8]) -> String {
    String::from_utf8_lossy(name).into_owned()
}

fn bytes_at<'a>(
    object: &'a [u8],
    offset: usize,
    size: usize,
    context: &'static str,
) -> Result<&'a [u8]> {
    let Some(end) = offset.checked_add(size) else {
        return Err(ElfTruncatedSnafu { context, offset }.build());
    };
    let Some(bytes) = object.get(offset..end) else {
        return Err(ElfTruncatedSnafu { context, offset }.build());
    };
    Ok(bytes)
}

fn read_u16(bytes: &[u8], offset: usize, context: &'static str) -> Result<u16> {
    let mut output = [0u8; 2];
    output.copy_from_slice(bytes_at(bytes, offset, 2, context)?);
    Ok(u16::from_le_bytes(output))
}

fn read_u32(bytes: &[u8], offset: usize, context: &'static str) -> Result<u32> {
    let mut output = [0u8; 4];
    output.copy_from_slice(bytes_at(bytes, offset, 4, context)?);
    Ok(u32::from_le_bytes(output))
}

fn read_u64(bytes: &[u8], offset: usize, context: &'static str) -> Result<u64> {
    let mut output = [0u8; 8];
    output.copy_from_slice(bytes_at(bytes, offset, 8, context)?);
    Ok(u64::from_le_bytes(output))
}

fn usize_from(value: u64, field: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        UnsupportedElfHeaderSnafu {
            field,
            actual: value,
            expected: u64::MAX,
        }
        .build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truncated_and_non_elf_objects_without_an_llvm_toolchain() {
        assert!(matches!(
            inspect_relocatable_text(&[]),
            Err(crate::Error::ElfTruncated { .. })
        ));
        assert!(matches!(
            inspect_relocatable_text(&[0; ELF_HEADER_BYTES]),
            Err(crate::Error::UnsupportedElfIdent { field: "magic", .. })
        ));
    }
}
