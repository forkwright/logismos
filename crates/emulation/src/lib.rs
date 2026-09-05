//! A deliberately small, CPU-only executor for verified gfx1100 instruction words.
//!
//! This crate is a test aid, not an HSA loader, code-object validator, or GPU
//! performance model. It accepts raw little-endian instruction words and rejects
//! every encoding other than the forms documented by [`SUPPORTED_INSTRUCTIONS`].
//! Normal emulator tests use checked instruction bytes and require no AMD toolchain.
//! The separately ignored `llvm_artifact` test rebuilds the pinned fixture when the
//! local AOMP witness is deliberately requested.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use snafu::Snafu;

/// Strict inspection of non-runnable AMDGPU relocatable text fixtures.
pub mod elf;

/// Number of lanes in the only supported wavefront shape.
pub const WAVE32_LANES: usize = 32;

/// Maximum number of vector registers held by a program state.
pub const MAX_VECTOR_REGISTERS: usize = 256;

/// Maximum raw text size accepted by one program.
pub const MAX_TEXT_BYTES: usize = 4096;

/// Returns the single configured GPU target from `contracts/gpu-target.txt`.
///
/// This is a function because trimming the newline in the contract is not a
/// stable constant operation on the fleet Rust toolchain.
#[must_use]
pub fn gfx1100_target() -> &'static str {
    include_str!("../../../contracts/gpu-target.txt").trim()
}

/// Exact instruction forms accepted by this executor.
pub const SUPPORTED_INSTRUCTIONS: [&str; 5] = [
    "v_mov_b32_e32",
    "v_add_nc_u32_e32",
    "v_add_f32_e32",
    "v_wmma_f32_16x16x16_f16 (exact-integer subset)",
    "s_endpgm",
];

/// Result type returned by the emulator.
pub type Result<T> = core::result::Result<T, Error>;

/// A validated raw gfx1100 instruction stream and its initial wave32 registers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wave32Program {
    text: Vec<u8>,
    initial_registers: Vec<[u32; WAVE32_LANES]>,
    max_steps: usize,
}

impl Wave32Program {
    /// Creates a bounded raw-instruction program.
    ///
    /// `text` must contain complete, four-byte little-endian instruction words.
    /// This constructor deliberately does not accept an ELF, AMDHSA metadata,
    /// relocations, memory, or floating-point mode state.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if text, the register file, or the step budget exceeds
    /// this slice's fixed bounds.
    pub fn new(
        text: Vec<u8>,
        initial_registers: Vec<[u32; WAVE32_LANES]>,
        max_steps: usize,
    ) -> Result<Self> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(TextTooLargeSnafu {
                actual: text.len(),
                maximum: MAX_TEXT_BYTES,
            }
            .build());
        }
        if !text.len().is_multiple_of(core::mem::size_of::<u32>()) {
            return Err(TruncatedInstructionSnafu { actual: text.len() }.build());
        }
        if initial_registers.len() > MAX_VECTOR_REGISTERS {
            return Err(RegisterFileTooLargeSnafu {
                actual: initial_registers.len(),
                maximum: MAX_VECTOR_REGISTERS,
            }
            .build());
        }
        if max_steps == 0 {
            return Err(ZeroStepBudgetSnafu.build());
        }

        Ok(Self {
            text,
            initial_registers,
            max_steps,
        })
    }

    /// Executes the instruction stream from its supplied register state.
    ///
    /// The f32 form admits only finite, normal inputs and a finite, normal host
    /// result. It uses host IEEE-754 `f32` addition as a CPU oracle within that
    /// narrow domain; it does not claim RDNA3 FP-mode, denormal, NaN, flag, or
    /// exception behavior. EXEC masking and source modifiers are also unsupported.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] for an unsupported encoding, operand, or floating-point
    /// class; an unavailable vector register; an exhausted budget; or text without
    /// `s_endpgm`.
    pub fn execute(&self) -> Result<ExecutionReport> {
        let mut registers = self.initial_registers.clone();
        let mut coverage = InstructionCoverage::default();
        let mut pc = 0usize;
        let mut steps = 0usize;

        loop {
            if steps == self.max_steps {
                coverage.refuse();
                return Err(ExecutionBudgetExceededSnafu {
                    maximum: self.max_steps,
                    coverage,
                }
                .build());
            }

            let instruction_pc = pc;
            let Some(encoded) = self.text.get(pc..pc.saturating_add(INSTRUCTION_BYTES)) else {
                coverage.refuse();
                return Err(MissingTerminatorSnafu { pc, coverage }.build());
            };
            let mut bytes = [0u8; INSTRUCTION_BYTES];
            bytes.copy_from_slice(encoded);
            let word = u32::from_le_bytes(bytes);
            let trailing = self
                .text
                .get(pc.saturating_add(INSTRUCTION_BYTES)..pc.saturating_add(WMMA_BYTES))
                .map(|encoded| {
                    let mut bytes = [0u8; INSTRUCTION_BYTES];
                    bytes.copy_from_slice(encoded);
                    u32::from_le_bytes(bytes)
                });
            let instruction = match decode(word, trailing) {
                Ok(instruction) => instruction,
                Err(error) => {
                    coverage.refuse();
                    return Err(error.with_coverage(word, instruction_pc, coverage));
                }
            };
            steps = steps.saturating_add(1);
            pc = pc.saturating_add(instruction.width());

            if apply_instruction(&mut registers, instruction, instruction_pc, &mut coverage)? {
                return Ok(ExecutionReport {
                    registers,
                    coverage,
                });
            }
        }
    }
}

/// Register state and instruction coverage from a completed execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReport {
    registers: Vec<[u32; WAVE32_LANES]>,
    coverage: InstructionCoverage,
}

impl ExecutionReport {
    /// Returns the complete final vector-register file.
    #[must_use]
    pub fn registers(&self) -> &[[u32; WAVE32_LANES]] {
        &self.registers
    }

    /// Returns the supported and refused instruction counts.
    #[must_use]
    pub const fn coverage(&self) -> InstructionCoverage {
        self.coverage
    }
}

/// Counts execution of this slice's instruction forms and refusals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InstructionCoverage {
    moves: usize,
    add_u32: usize,
    add_f32: usize,
    wmma_f32: usize,
    end_program: usize,
    refusals: usize,
}

impl InstructionCoverage {
    /// Returns the number of executed `v_mov_b32_e32` words.
    #[must_use]
    pub const fn move_count(self) -> usize {
        self.moves
    }

    /// Returns the number of executed `v_add_nc_u32_e32` words.
    #[must_use]
    pub const fn add_u32_count(self) -> usize {
        self.add_u32
    }

    /// Returns the number of executed `v_add_f32_e32` words.
    #[must_use]
    pub const fn add_f32_count(self) -> usize {
        self.add_f32
    }

    /// Returns the number of executed exact-integer WMMA words.
    #[must_use]
    pub const fn wmma_f32_count(self) -> usize {
        self.wmma_f32
    }

    /// Returns the number of executed `s_endpgm` words.
    #[must_use]
    pub const fn end_program_count(self) -> usize {
        self.end_program
    }

    /// Returns the number of rejected decode or execution attempts.
    #[must_use]
    pub const fn refused_count(self) -> usize {
        self.refusals
    }

    fn refuse(&mut self) {
        self.refusals = self.refusals.saturating_add(1);
    }
}

/// Typed errors for invalid input and unsupported execution.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Raw text exceeded the fixed executor bound.
    #[snafu(display("raw instruction text is {actual} bytes; maximum is {maximum}"))]
    TextTooLarge {
        /// Input text length.
        actual: usize,
        /// Largest supported input text length.
        maximum: usize,
    },

    /// Raw text ended in a partial instruction word.
    #[snafu(display("raw instruction text is {actual} bytes, not a multiple of four"))]
    TruncatedInstruction {
        /// Input text length.
        actual: usize,
    },

    /// Initial state contained more registers than the native field can address.
    #[snafu(display("register file has {actual} entries; maximum is {maximum}"))]
    RegisterFileTooLarge {
        /// Input register count.
        actual: usize,
        /// Largest supported input register count.
        maximum: usize,
    },

    /// Program supplied no execution budget.
    #[snafu(display("execution budget must be nonzero"))]
    ZeroStepBudget,

    /// Word did not match one of the exact accepted instruction encodings.
    #[snafu(display("unsupported gfx1100 instruction encoding {word:#010x} at byte {pc}"))]
    UnsupportedEncoding {
        /// Native instruction word.
        word: u32,
        /// Byte offset of the word.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// Word requested a non-VGPR source such as a literal or scalar operand.
    #[snafu(display("unsupported source operand {encoded:#05x} in {word:#010x} at byte {pc}"))]
    UnsupportedSourceOperand {
        /// Raw nine-bit AMDGPU source encoding.
        encoded: u16,
        /// Native instruction word.
        word: u32,
        /// Byte offset of the word.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// Floating-point input or result was outside the executable host-oracle domain.
    #[snafu(display("unsupported f32 {operand} class {bits:#010x} at byte {pc}"))]
    UnsupportedF32Class {
        /// Whether the rejected value was a source or result.
        operand: &'static str,
        /// IEEE-754 bit pattern.
        bits: u32,
        /// Byte offset of the instruction.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// WMMA register block was unaligned, unavailable, or overlaps another operand.
    #[snafu(display("unsupported WMMA {operand} register block v{base}: {reason} at byte {pc}"))]
    UnsupportedWmmaRegisters {
        /// Operand name.
        operand: &'static str,
        /// First VGPR in the block.
        base: u8,
        /// Fixed rejection reason.
        reason: &'static str,
        /// Instruction byte offset.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// WMMA value was outside the exact-integer subset.
    #[snafu(display("unsupported WMMA {operand} value {bits:#010x} at byte {pc}"))]
    UnsupportedWmmaValue {
        /// Operand name.
        operand: &'static str,
        /// Source bit pattern.
        bits: u32,
        /// Instruction byte offset.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// WMMA A/B second half did not replicate the first half-wave.
    #[snafu(display("WMMA {operand} v{register} lane {lane} is not replicated at byte {pc}"))]
    WmmaReplicationMismatch {
        /// Operand name.
        operand: &'static str,
        /// VGPR number.
        register: u8,
        /// First-half lane number.
        lane: usize,
        /// Instruction byte offset.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// Decoded vector register was outside the supplied register file.
    #[snafu(display("vector register v{register} unavailable at byte {pc}"))]
    RegisterOutOfBounds {
        /// Decoded VGPR number.
        register: u8,
        /// Byte offset of the instruction.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// Execution consumed the caller-provided instruction budget.
    #[snafu(display("execution exceeded the {maximum}-instruction budget"))]
    ExecutionBudgetExceeded {
        /// Maximum permitted decoded words.
        maximum: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// Text reached its end without the required terminating instruction.
    #[snafu(display("raw instruction text ended without s_endpgm at byte {pc}"))]
    MissingTerminator {
        /// Byte offset immediately after the last decoded word.
        pc: usize,
        /// Coverage up to and including this refusal.
        coverage: InstructionCoverage,
    },

    /// ELF object exceeded the bounded inspection input size.
    #[snafu(display("ELF object is {actual} bytes; maximum is {maximum}"))]
    ElfTooLarge {
        /// Input object length.
        actual: usize,
        /// Largest supported object length.
        maximum: usize,
    },

    /// ELF object was too short to read a required region.
    #[snafu(display("ELF object is truncated while reading {context} at byte {offset}"))]
    ElfTruncated {
        /// Human-readable fixed parser context.
        context: &'static str,
        /// Required byte offset.
        offset: usize,
    },

    /// ELF identification byte did not match the relocatable fixture contract.
    #[snafu(display("unsupported ELF {field}: got {actual:#04x}, expected {expected:#04x}"))]
    UnsupportedElfIdent {
        /// ELF identification field name.
        field: &'static str,
        /// Observed byte.
        actual: u8,
        /// Required byte.
        expected: u8,
    },

    /// ELF header field did not match the relocatable fixture contract.
    #[snafu(display("unsupported ELF {field}: got {actual:#x}, expected {expected:#x}"))]
    UnsupportedElfHeader {
        /// ELF header field name.
        field: &'static str,
        /// Observed numeric value.
        actual: u64,
        /// Required numeric value.
        expected: u64,
    },

    /// ELF contains program headers and is not an inspectable relocatable fixture.
    #[snafu(display("ELF program-header count {count} is unsupported for a relocatable fixture"))]
    ProgramHeadersUnsupported {
        /// Observed program-header count.
        count: u16,
    },

    /// ELF contains a relocation section, which this no-memory executor cannot apply.
    #[snafu(display("ELF relocation section {section} is unsupported"))]
    RelocationsUnsupported {
        /// Relocation section name.
        section: String,
    },

    /// ELF section is not part of the narrow relocatable-text fixture contract.
    #[snafu(display("ELF section {section} (type {section_type:#x}) is unsupported"))]
    UnsupportedElfSection {
        /// Section name.
        section: String,
        /// ELF section type.
        section_type: u32,
    },

    /// ELF fixture had no unique executable `.text` section.
    #[snafu(display("ELF relocatable fixture is missing executable .text"))]
    MissingExecutableText,

    /// ELF `.text` was malformed for raw wave32 instruction dispatch.
    #[snafu(display("ELF .text is unsupported: {reason}"))]
    UnsupportedTextSection {
        /// Reason derived by this narrow parser.
        reason: &'static str,
    },
}

impl Error {
    /// Returns the coverage record carried by an execution error, if applicable.
    #[must_use]
    pub const fn coverage(&self) -> Option<InstructionCoverage> {
        match self {
            Self::UnsupportedEncoding { coverage, .. }
            | Self::UnsupportedSourceOperand { coverage, .. }
            | Self::UnsupportedF32Class { coverage, .. }
            | Self::UnsupportedWmmaRegisters { coverage, .. }
            | Self::UnsupportedWmmaValue { coverage, .. }
            | Self::WmmaReplicationMismatch { coverage, .. }
            | Self::RegisterOutOfBounds { coverage, .. }
            | Self::ExecutionBudgetExceeded { coverage, .. }
            | Self::MissingTerminator { coverage, .. } => Some(*coverage),
            Self::TextTooLarge { .. }
            | Self::TruncatedInstruction { .. }
            | Self::RegisterFileTooLarge { .. }
            | Self::ZeroStepBudget
            | Self::ElfTooLarge { .. }
            | Self::ElfTruncated { .. }
            | Self::UnsupportedElfIdent { .. }
            | Self::UnsupportedElfHeader { .. }
            | Self::ProgramHeadersUnsupported { .. }
            | Self::RelocationsUnsupported { .. }
            | Self::UnsupportedElfSection { .. }
            | Self::MissingExecutableText
            | Self::UnsupportedTextSection { .. } => None,
        }
    }
}

const INSTRUCTION_BYTES: usize = 4;
const WMMA_BYTES: usize = 8;
const VGPR_SOURCE_BASE: u16 = 256;
const VGPR_SOURCE_LIMIT: u16 = VGPR_SOURCE_BASE + 256;
const V_MOV_B32_E32_MASK: u32 = 0xfe01_fe00;
const V_MOV_B32_E32_BASE: u32 = 0x7e00_0200;
const V_ADD_NC_U32_E32_MASK: u32 = 0xfe00_0000;
const V_ADD_NC_U32_E32_BASE: u32 = 0x4a00_0000;
const V_ADD_F32_E32_MASK: u32 = 0xfe00_0000;
const V_ADD_F32_E32_BASE: u32 = 0x0600_0000;
const S_ENDPGM: u32 = 0xbfb0_0000;
const V_WMMA_F32_WORD0_MASK: u32 = 0xffff_ff00;
const V_WMMA_F32_WORD0_BASE: u32 = 0xcc40_4000;
const V_WMMA_F32_WORD1_MASK: u32 = 0xfc00_0000;
const V_WMMA_F32_WORD1_BASE: u32 = 0x1c00_0000;

#[derive(Debug, Clone, Copy)]
enum Instruction {
    Move {
        destination: u8,
        source: u8,
    },
    AddU32 {
        destination: u8,
        source0: u8,
        source1: u8,
    },
    AddF32 {
        destination: u8,
        source0: u8,
        source1: u8,
    },
    WmmaF32 {
        destination: u8,
        a: u8,
        b: u8,
        c: u8,
    },
    EndProgram,
}

impl Instruction {
    const fn width(self) -> usize {
        match self {
            Self::WmmaF32 { .. } => WMMA_BYTES,
            Self::Move { .. } | Self::AddU32 { .. } | Self::AddF32 { .. } | Self::EndProgram => {
                INSTRUCTION_BYTES
            }
        }
    }
}

enum DecodeError {
    Encoding,
    Source { encoded: u16 },
}

impl DecodeError {
    fn with_coverage(self, word: u32, pc: usize, coverage: InstructionCoverage) -> Error {
        match self {
            Self::Encoding => UnsupportedEncodingSnafu { word, pc, coverage }.build(),
            Self::Source { encoded } => UnsupportedSourceOperandSnafu {
                encoded,
                word,
                pc,
                coverage,
            }
            .build(),
        }
    }
}

fn decode(word: u32, trailing: Option<u32>) -> core::result::Result<Instruction, DecodeError> {
    if word == S_ENDPGM {
        return Ok(Instruction::EndProgram);
    }

    if word & V_MOV_B32_E32_MASK == V_MOV_B32_E32_BASE {
        return Ok(Instruction::Move {
            destination: byte_at(word, 17),
            source: decode_vgpr_source(source0(word))?,
        });
    }
    if word & V_ADD_NC_U32_E32_MASK == V_ADD_NC_U32_E32_BASE {
        return Ok(Instruction::AddU32 {
            destination: byte_at(word, 17),
            source0: decode_vgpr_source(source0(word))?,
            source1: byte_at(word, 9),
        });
    }
    if word & V_ADD_F32_E32_MASK == V_ADD_F32_E32_BASE {
        return Ok(Instruction::AddF32 {
            destination: byte_at(word, 17),
            source0: decode_vgpr_source(source0(word))?,
            source1: byte_at(word, 9),
        });
    }
    if word & V_WMMA_F32_WORD0_MASK == V_WMMA_F32_WORD0_BASE {
        let Some(word1) = trailing else {
            return Err(DecodeError::Encoding);
        };
        if word1 & V_WMMA_F32_WORD1_MASK != V_WMMA_F32_WORD1_BASE {
            return Err(DecodeError::Encoding);
        }
        return Ok(Instruction::WmmaF32 {
            destination: byte_at(word, 0),
            a: decode_vgpr_source(field9(word1, 0))?,
            b: decode_vgpr_source(field9(word1, 9))?,
            c: byte_at(word1, 18),
        });
    }

    Err(DecodeError::Encoding)
}

fn field9(word: u32, shift: u32) -> u16 {
    let shifted = word >> shift;
    let bytes = shifted.to_le_bytes();
    u16::from_le_bytes([bytes[0], bytes[1] & 1])
}

fn byte_at(word: u32, shift: u32) -> u8 {
    (word >> shift).to_le_bytes()[0]
}

fn source0(word: u32) -> u16 {
    let bytes = word.to_le_bytes();
    u16::from_le_bytes([bytes[0], bytes[1] & 1])
}

fn decode_vgpr_source(encoded: u16) -> core::result::Result<u8, DecodeError> {
    if !(VGPR_SOURCE_BASE..VGPR_SOURCE_LIMIT).contains(&encoded) {
        return Err(DecodeError::Source { encoded });
    }
    Ok((encoded - VGPR_SOURCE_BASE).to_le_bytes()[0])
}

fn apply_instruction(
    registers: &mut [[u32; WAVE32_LANES]],
    instruction: Instruction,
    pc: usize,
    coverage: &mut InstructionCoverage,
) -> Result<bool> {
    match instruction {
        Instruction::Move {
            destination,
            source,
        } => {
            let value = read_register(registers, source, pc, *coverage)?;
            write_register(registers, destination, value, pc, *coverage)?;
            coverage.moves = coverage.moves.saturating_add(1);
        }
        Instruction::AddU32 {
            destination,
            source0,
            source1,
        } => {
            let left = read_register(registers, source0, pc, *coverage)?;
            let right = read_register(registers, source1, pc, *coverage)?;
            let result = lane_binary(left, right, u32::wrapping_add);
            write_register(registers, destination, result, pc, *coverage)?;
            coverage.add_u32 = coverage.add_u32.saturating_add(1);
        }
        Instruction::AddF32 {
            destination,
            source0,
            source1,
        } => {
            let left = read_register(registers, source0, pc, *coverage)?;
            let right = read_register(registers, source1, pc, *coverage)?;
            let result = lane_f32_add(left, right, pc, *coverage)?;
            write_register(registers, destination, result, pc, *coverage)?;
            coverage.add_f32 = coverage.add_f32.saturating_add(1);
        }
        Instruction::WmmaF32 {
            destination,
            a,
            b,
            c,
        } => {
            apply_wmma_f32(registers, destination, a, b, c, pc, *coverage)?;
            coverage.wmma_f32 = coverage.wmma_f32.saturating_add(1);
        }
        Instruction::EndProgram => {
            coverage.end_program = coverage.end_program.saturating_add(1);
            return Ok(true);
        }
    }
    Ok(false)
}

// AMD RDNA3 ISA 70650 (Feb. 2023), §7.9/Table 33 and p.75 define this
// VOP3P form and its wave32 A/B replication plus A/B/C/D VGPR layouts.
// LLVM AMDGPUAsmGFX11 (18.1.7) documents the operand shape. The local LLVM
// witness in tests/fixtures records the exact accepted gfx1100 bytes.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "the ISA's four register operands and 16x16 lane mapping are kept together for auditability"
)]
fn apply_wmma_f32(
    registers: &mut [[u32; WAVE32_LANES]],
    destination: u8,
    a: u8,
    b: u8,
    c: u8,
    pc: usize,
    coverage: InstructionCoverage,
) -> Result<()> {
    for (operand, base) in [("destination", destination), ("A", a), ("B", b), ("C", c)] {
        ensure_wmma_block(registers, operand, base, pc, coverage)?;
    }
    for (left, right) in [(destination, a), (destination, b), (destination, c)] {
        if blocks_overlap(left, right) {
            return wmma_register_error(
                "destination",
                destination,
                "overlaps source",
                pc,
                coverage,
            );
        }
    }
    let mut matrix_a = [[0i32; 16]; 16];
    let mut matrix_b = [[0i32; 16]; 16];
    let mut matrix_c = [[0i32; 16]; 16];
    for row in 0..16 {
        for packed in 0u8..8 {
            let register = a.saturating_add(packed);
            let value = registers[usize::from(register)][row];
            if value != registers[usize::from(register)][row + 16] {
                let mut refused = coverage;
                refused.refuse();
                return Err(WmmaReplicationMismatchSnafu {
                    operand: "A",
                    register,
                    lane: row,
                    pc,
                    coverage: refused,
                }
                .build());
            }
            let bytes = value.to_le_bytes();
            let index = usize::from(packed) * 2;
            matrix_a[row][index] =
                exact_f16(u16::from_le_bytes([bytes[0], bytes[1]]), "A", pc, coverage)?;
            matrix_a[row][index + 1] =
                exact_f16(u16::from_le_bytes([bytes[2], bytes[3]]), "A", pc, coverage)?;
        }
    }
    for column in 0..16 {
        for packed in 0u8..8 {
            let register = b.saturating_add(packed);
            let value = registers[usize::from(register)][column];
            if value != registers[usize::from(register)][column + 16] {
                let mut refused = coverage;
                refused.refuse();
                return Err(WmmaReplicationMismatchSnafu {
                    operand: "B",
                    register,
                    lane: column,
                    pc,
                    coverage: refused,
                }
                .build());
            }
            let bytes = value.to_le_bytes();
            let index = usize::from(packed) * 2;
            matrix_b[index][column] =
                exact_f16(u16::from_le_bytes([bytes[0], bytes[1]]), "B", pc, coverage)?;
            matrix_b[index + 1][column] =
                exact_f16(u16::from_le_bytes([bytes[2], bytes[3]]), "B", pc, coverage)?;
        }
    }
    for row in 0..16 {
        for column in 0..16 {
            let lane = (row % 2) * 16 + column;
            matrix_c[row][column] =
                exact_f32_integer(registers[usize::from(c) + row / 2][lane], "C", pc, coverage)?;
        }
    }
    for row in 0..16 {
        for column in 0..16 {
            let mut sum = i64::from(matrix_c[row][column]);
            for k in 0..16 {
                sum = sum
                    .checked_add(i64::from(matrix_a[row][k]) * i64::from(matrix_b[k][column]))
                    .ok_or_else(|| wmma_value_error("accumulator", 0, pc, coverage))?;
            }
            let result =
                i32::try_from(sum).map_err(|_| wmma_value_error("accumulator", 0, pc, coverage))?;
            let lane = (row % 2) * 16 + column;
            registers[usize::from(destination) + row / 2][lane] = encode_f32_integer(result);
        }
    }
    Ok(())
}

fn ensure_wmma_block(
    registers: &[[u32; WAVE32_LANES]],
    operand: &'static str,
    base: u8,
    pc: usize,
    coverage: InstructionCoverage,
) -> Result<()> {
    if !base.is_multiple_of(8) || usize::from(base).saturating_add(8) > registers.len() {
        return wmma_register_error(
            operand,
            base,
            "requires eight aligned available VGPRs",
            pc,
            coverage,
        );
    }
    Ok(())
}
fn blocks_overlap(left: u8, right: u8) -> bool {
    left == right
}
fn wmma_register_error(
    operand: &'static str,
    base: u8,
    reason: &'static str,
    pc: usize,
    mut coverage: InstructionCoverage,
) -> Result<()> {
    coverage.refuse();
    Err(UnsupportedWmmaRegistersSnafu {
        operand,
        base,
        reason,
        pc,
        coverage,
    }
    .build())
}
fn wmma_value_error(
    operand: &'static str,
    bits: u32,
    pc: usize,
    mut coverage: InstructionCoverage,
) -> Error {
    coverage.refuse();
    UnsupportedWmmaValueSnafu {
        operand,
        bits,
        pc,
        coverage,
    }
    .build()
}
fn exact_f16(
    bits: u16,
    operand: &'static str,
    pc: usize,
    coverage: InstructionCoverage,
) -> Result<i32> {
    if bits == 0 {
        return Ok(0);
    }
    let sign = if bits & 0x8000 == 0 { 1 } else { -1 };
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    if exponent == 0 || exponent == 0x1f {
        return Err(wmma_value_error(operand, u32::from(bits), pc, coverage));
    }
    let shift = i32::from(exponent) - 25;
    let significand = i32::from(1024 + fraction);
    let magnitude = if shift >= 0 {
        significand
            .checked_shl(u32::try_from(shift).unwrap_or(32))
            .unwrap_or(0)
    } else {
        let divisor = 1_i32
            .checked_shl(u32::try_from(-shift).unwrap_or(32))
            .unwrap_or(0);
        if divisor == 0 || significand % divisor != 0 {
            return Err(wmma_value_error(operand, u32::from(bits), pc, coverage));
        }
        significand / divisor
    };
    if magnitude > 64 {
        return Err(wmma_value_error(operand, u32::from(bits), pc, coverage));
    }
    Ok(sign * magnitude)
}
fn exact_f32_integer(
    bits: u32,
    operand: &'static str,
    pc: usize,
    coverage: InstructionCoverage,
) -> Result<i32> {
    if bits == 0 || bits == 0x8000_0000 {
        return Ok(0);
    }
    let sign = if bits & 0x8000_0000 == 0 { 1 } else { -1 };
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    if exponent == 0 || exponent == 0xff {
        return Err(wmma_value_error(operand, bits, pc, coverage));
    }
    let shift = i32::try_from(exponent).unwrap_or(0) - 150;
    let significand = i64::from(0x80_0000 | fraction);
    let magnitude = if shift >= 0 {
        significand
            .checked_shl(u32::try_from(shift).unwrap_or(64))
            .unwrap_or(0)
    } else {
        let divisor = 1_i64
            .checked_shl(u32::try_from(-shift).unwrap_or(64))
            .unwrap_or(0);
        if divisor == 0 || significand % divisor != 0 {
            return Err(wmma_value_error(operand, bits, pc, coverage));
        }
        significand / divisor
    };
    if magnitude > 131_072 {
        return Err(wmma_value_error(operand, bits, pc, coverage));
    }
    let magnitude =
        i32::try_from(magnitude).map_err(|_| wmma_value_error(operand, bits, pc, coverage))?;
    Ok(sign * magnitude)
}
fn encode_f32_integer(value: i32) -> u32 {
    if value == 0 {
        return 0;
    }
    let sign = if value < 0 { 0x8000_0000 } else { 0 };
    let magnitude = value.unsigned_abs();
    let highest = magnitude.ilog2();
    let exponent = highest + 127;
    let fraction = (magnitude - (1_u32 << highest)) << (23 - highest);
    sign | (exponent << 23) | fraction
}

fn lane_f32_add(
    left: [u32; WAVE32_LANES],
    right: [u32; WAVE32_LANES],
    pc: usize,
    coverage: InstructionCoverage,
) -> Result<[u32; WAVE32_LANES]> {
    let mut output = [0u32; WAVE32_LANES];
    for ((destination, left), right) in output.iter_mut().zip(left).zip(right) {
        ensure_normal_f32(left, "source0", pc, coverage)?;
        ensure_normal_f32(right, "source1", pc, coverage)?;
        let result = (f32::from_bits(left) + f32::from_bits(right)).to_bits();
        ensure_normal_f32(result, "result", pc, coverage)?;
        *destination = result;
    }
    Ok(output)
}

fn ensure_normal_f32(
    bits: u32,
    operand: &'static str,
    pc: usize,
    mut coverage: InstructionCoverage,
) -> Result<()> {
    let exponent = bits & 0x7f80_0000;
    if exponent != 0 && exponent != 0x7f80_0000 {
        return Ok(());
    }
    coverage.refuse();
    Err(UnsupportedF32ClassSnafu {
        operand,
        bits,
        pc,
        coverage,
    }
    .build())
}

fn read_register(
    registers: &[[u32; WAVE32_LANES]],
    register: u8,
    pc: usize,
    mut coverage: InstructionCoverage,
) -> Result<[u32; WAVE32_LANES]> {
    let Some(value) = registers.get(usize::from(register)) else {
        coverage.refuse();
        return Err(RegisterOutOfBoundsSnafu {
            register,
            pc,
            coverage,
        }
        .build());
    };
    Ok(*value)
}

fn write_register(
    registers: &mut [[u32; WAVE32_LANES]],
    register: u8,
    value: [u32; WAVE32_LANES],
    pc: usize,
    mut coverage: InstructionCoverage,
) -> Result<()> {
    let Some(destination) = registers.get_mut(usize::from(register)) else {
        coverage.refuse();
        return Err(RegisterOutOfBoundsSnafu {
            register,
            pc,
            coverage,
        }
        .build());
    };
    *destination = value;
    Ok(())
}

fn lane_binary(
    left: [u32; WAVE32_LANES],
    right: [u32; WAVE32_LANES],
    operation: impl Fn(u32, u32) -> u32,
) -> [u32; WAVE32_LANES] {
    let mut output = [0u32; WAVE32_LANES];
    for ((destination, left), right) in output.iter_mut().zip(left).zip(right) {
        *destination = operation(left, right);
    }
    output
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::needless_range_loop,
    reason = "fixtures use expect to name invariants and explicit indices to state the AMD lane mapping"
)]
mod tests {
    use super::*;

    const COPY_ADD: [u8; 12] = [
        0x00, 0x03, 0x02, 0x7e, // v_mov_b32_e32 v1, v0
        0x00, 0x03, 0x04, 0x4a, // v_add_nc_u32_e32 v2, v0, v1
        0x00, 0x00, 0xb0, 0xbf, // s_endpgm
    ];

    #[test]
    fn executes_native_copy_and_wrapping_add_in_every_wave32_lane() {
        let mut input = [0u32; WAVE32_LANES];
        for (lane, value) in input.iter_mut().enumerate() {
            *value = u32::from(lane.to_le_bytes()[0]).wrapping_add(u32::MAX - 30);
        }
        let program = Wave32Program::new(COPY_ADD.to_vec(), vec![input, [0; 32], [0; 32]], 3)
            .expect("fixture is bounded and aligned");

        let report = program.execute().expect("native fixture is supported");
        assert_eq!(report.registers()[1], input);
        for (lane, value) in report.registers()[2].iter().enumerate() {
            assert_eq!(*value, input[lane].wrapping_add(input[lane]));
        }
        assert_eq!(report.coverage().move_count(), 1);
        assert_eq!(report.coverage().add_u32_count(), 1);
        assert_eq!(report.coverage().end_program_count(), 1);
    }

    #[test]
    fn executes_native_f32_add_from_llvm_verified_word() {
        let text = [
            0x00, 0x03, 0x06, 0x06, // v_add_f32_e32 v3, v0, v1
            0x00, 0x00, 0xb0, 0xbf,
        ];
        let mut left = [0u32; WAVE32_LANES];
        let mut right = [0u32; WAVE32_LANES];
        for (lane, value) in left.iter_mut().enumerate() {
            *value = (f32::from(lane.to_le_bytes()[0]) + 1.0).to_bits();
        }
        for (lane, value) in right.iter_mut().enumerate() {
            *value = (f32::from(lane.to_le_bytes()[0]) + 2.0).to_bits();
        }
        let program = Wave32Program::new(text.to_vec(), vec![left, right, [0; 32], [0; 32]], 2)
            .expect("fixture is bounded and aligned");

        let report = program.execute().expect("native fixture is supported");
        for (lane, value) in report.registers()[3].iter().enumerate() {
            let expected = f32::from_bits(left[lane]) + f32::from_bits(right[lane]);
            assert_eq!(*value, expected.to_bits());
        }
        assert_eq!(report.coverage().add_f32_count(), 1);
    }

    #[test]
    fn rejects_subnormal_sources_and_non_normal_f32_results() {
        let text = [
            0x00, 0x03, 0x06, 0x06, // v_add_f32_e32 v3, v0, v1
            0x00, 0x00, 0xb0, 0xbf,
        ];
        let subnormal = Wave32Program::new(
            text.to_vec(),
            vec![[1; 32], [1.0f32.to_bits(); 32], [0; 32], [0; 32]],
            2,
        )
        .expect("fixture is bounded and aligned")
        .execute()
        .expect_err("subnormal source is outside the host-float contract");
        assert!(matches!(
            subnormal,
            Error::UnsupportedF32Class {
                operand: "source0",
                ..
            }
        ));

        let infinity = Wave32Program::new(
            text.to_vec(),
            vec![
                [f32::MAX.to_bits(); 32],
                [f32::MAX.to_bits(); 32],
                [0; 32],
                [0; 32],
            ],
            2,
        )
        .expect("fixture is bounded and aligned")
        .execute()
        .expect_err("infinite result is outside the host-float contract");
        assert!(matches!(
            infinity,
            Error::UnsupportedF32Class {
                operand: "result",
                ..
            }
        ));
    }

    #[test]
    fn reads_the_target_from_the_contract_single_source_of_truth() {
        assert_eq!(gfx1100_target(), "gfx1100");
    }

    #[test]
    fn rejects_literals_instead_of_treating_them_as_registers() {
        let text = [
            0xf2, 0x04, 0x02, 0x06, // v_add_f32_e32 v1, 1.0, v2
            0x00, 0x00, 0xb0, 0xbf,
        ];
        let program = Wave32Program::new(text.to_vec(), vec![[0; 32]; 3], 2)
            .expect("fixture is bounded and aligned");

        let error = program
            .execute()
            .expect_err("literal source is unsupported");
        assert!(matches!(
            error,
            Error::UnsupportedSourceOperand { encoded: 0xf2, .. }
        ));
        assert_eq!(
            error.coverage().expect("execution error").refused_count(),
            1
        );
    }

    #[test]
    fn rejects_unknown_encoding_and_reports_refusal() {
        let program = Wave32Program::new(0u32.to_le_bytes().to_vec(), vec![[0; 32]], 1)
            .expect("fixture is bounded and aligned");

        let error = program.execute().expect_err("zero word is unsupported");
        assert!(matches!(error, Error::UnsupportedEncoding { .. }));
        assert_eq!(
            error.coverage().expect("execution error").refused_count(),
            1
        );
    }

    #[test]
    fn rejects_missing_register_budget_and_partial_word() {
        let register = Wave32Program::new(COPY_ADD.to_vec(), vec![[0; 32]], 3)
            .expect("fixture is bounded and aligned")
            .execute()
            .expect_err("v1 is not supplied");
        assert!(matches!(
            register,
            Error::RegisterOutOfBounds { register: 1, .. }
        ));

        let budget = Wave32Program::new(COPY_ADD.to_vec(), vec![[0; 32]; 3], 2)
            .expect("fixture is bounded and aligned")
            .execute()
            .expect_err("three native words exceed budget two");
        assert!(matches!(budget, Error::ExecutionBudgetExceeded { .. }));

        let truncated = Wave32Program::new(vec![0, 1, 2], vec![], 1)
            .expect_err("partial native word is rejected");
        assert!(matches!(truncated, Error::TruncatedInstruction { .. }));
    }

    #[test]
    fn refuses_text_without_a_native_terminator() {
        let program = Wave32Program::new(COPY_ADD[..8].to_vec(), vec![[0; 32]; 3], 3)
            .expect("fixture is bounded and aligned");

        let error = program
            .execute()
            .expect_err("instruction stream must end in s_endpgm");
        assert!(matches!(error, Error::MissingTerminator { pc: 8, .. }));
        let coverage = error.coverage().expect("execution error");
        assert_eq!(coverage.move_count(), 1);
        assert_eq!(coverage.add_u32_count(), 1);
        assert_eq!(coverage.refused_count(), 1);
    }

    #[test]
    fn wmma_rejects_an_unpaired_eight_byte_encoding() {
        let text = [0x18, 0x40, 0x40, 0xcc];
        let result = Wave32Program::new(text.to_vec(), vec![[0; 32]; 32], 1)
            .expect("word aligned")
            .execute();
        assert!(matches!(result, Err(Error::UnsupportedEncoding { .. })));
    }

    const WMMA: [u8; 12] = [
        0x18, 0x40, 0x40, 0xcc, 0x00, 0x11, 0x42, 0x1c, 0, 0, 0xb0, 0xbf,
    ];
    fn half(value: i32) -> u16 {
        [0u16, 0x3c00, 0x4000, 0x4200, 0x4400][usize::try_from(value).unwrap_or(0)]
    }
    fn put_half(r: &mut [[u32; 32]], base: usize, lane: usize, index: usize, value: i32) {
        let shift = if index.is_multiple_of(2) { 0 } else { 16 };
        r[base + index / 2][lane] |= u32::from(half(value)) << shift;
    }
    #[test]
    fn wmma_basis_and_asymmetric_layout_oracles() {
        let mut r = vec![[0; 32]; 32];
        for lane in [2usize, 18] {
            put_half(&mut r, 0, lane, 3, 1);
        }
        for lane in [5usize, 21] {
            put_half(&mut r, 8, lane, 3, 1);
        }
        let d = Wave32Program::new(WMMA.to_vec(), r, 2)
            .expect("bounded")
            .execute()
            .expect("basis");
        assert_eq!(d.registers()[25][5], 1.0f32.to_bits());
        let mut r = vec![[0; 32]; 32];
        for row in 0..16 {
            for lane in [row, row + 16] {
                put_half(&mut r, 0, lane, row, 1);
            }
        }
        for k in 0..16 {
            for col in 0..16 {
                let v = i32::try_from((k * 3 + col * 5) % 4).unwrap_or(0);
                for lane in [col, col + 16] {
                    put_half(&mut r, 8, lane, k, v);
                }
            }
        }
        let d = Wave32Program::new(WMMA.to_vec(), r, 2)
            .expect("bounded")
            .execute()
            .expect("asymmetric");
        for row in 0..16 {
            for col in 0..16 {
                let v = i32::try_from((row * 3 + col * 5) % 4).unwrap_or(0);
                assert_eq!(
                    d.registers()[24 + row / 2][(row % 2) * 16 + col],
                    encode_f32_integer(v)
                );
            }
        }
        let mut r = vec![[0; 32]; 32];
        let mut a = [[0i32; 16]; 16];
        let mut b = [[0i32; 16]; 16];
        for row in 0..16 {
            for k in 0..16 {
                a[row][k] = i32::try_from((row + 2 * k) % 4).unwrap_or(0);
                for lane in [row, row + 16] {
                    put_half(&mut r, 0, lane, k, a[row][k]);
                }
            }
        }
        for k in 0..16 {
            for col in 0..16 {
                b[k][col] = i32::try_from((3 * k + col) % 4).unwrap_or(0);
                for lane in [col, col + 16] {
                    put_half(&mut r, 8, lane, k, b[k][col]);
                }
            }
        }
        let d = Wave32Program::new(WMMA.to_vec(), r, 2)
            .expect("bounded")
            .execute()
            .expect("dense");
        for row in 0..16 {
            for col in 0..16 {
                let mut expected = 0i32;
                for k in 0..16 {
                    expected += a[row][k] * b[k][col];
                }
                assert_eq!(
                    d.registers()[24 + row / 2][(row % 2) * 16 + col],
                    encode_f32_integer(expected)
                );
            }
        }
    }

    #[test]
    fn wmma_rejects_replication_and_destination_overlap() {
        let mut r = vec![[0; 32]; 32];
        put_half(&mut r, 0, 0, 0, 1);
        assert!(matches!(
            Wave32Program::new(WMMA.to_vec(), r, 2)
                .expect("bounded")
                .execute(),
            Err(Error::WmmaReplicationMismatch { .. })
        ));
        let overlap = [
            0x10, 0x40, 0x40, 0xcc, 0x00, 0x11, 0x42, 0x1c, 0, 0, 0xb0, 0xbf,
        ];
        assert!(matches!(
            Wave32Program::new(overlap.to_vec(), vec![[0; 32]; 32], 2)
                .expect("bounded")
                .execute(),
            Err(Error::UnsupportedWmmaRegisters { .. })
        ));
    }
}
