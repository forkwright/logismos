//! A deliberately small, CPU-only executor for verified gfx1100 instruction words.
//!
//! This crate is a test aid, not an HSA loader, code-object validator, or GPU
//! performance model. It accepts raw little-endian instruction words and rejects
//! every encoding other than the four forms documented by [`SUPPORTED_INSTRUCTIONS`].
//! Normal emulator tests use checked instruction bytes and require no AMD toolchain.
//! The separately ignored `llvm_artifact` test rebuilds the pinned fixture when the
//! local AOMP witness is deliberately requested.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use snafu::Snafu;

/// Number of lanes in the only supported wavefront shape.
pub const WAVE32_LANES: usize = 32;

/// Maximum number of vector registers held by a program state.
pub const MAX_VECTOR_REGISTERS: usize = 256;

/// Maximum raw text size accepted by one program.
pub const MAX_TEXT_BYTES: usize = 4096;

/// Target accepted by the artifact witness and this decoder.
pub const GFX1100_TARGET: &str = "gfx1100";

/// Exact instruction forms accepted by this executor.
pub const SUPPORTED_INSTRUCTIONS: [&str; 4] = [
    "v_mov_b32_e32",
    "v_add_nc_u32_e32",
    "v_add_f32_e32",
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
    /// The f32 form performs host IEEE-754 `f32` addition on lane bit patterns.
    /// It does not model hardware FP-mode registers, denormal behavior, status
    /// flags, EXEC masking, or any source modifiers; instructions that would set
    /// such state are unsupported encodings.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] for an unsupported encoding or operand, an unavailable
    /// vector register, an exhausted budget, or text without `s_endpgm`.
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
            let instruction = match decode(word) {
                Ok(instruction) => instruction,
                Err(error) => {
                    coverage.refuse();
                    return Err(error.with_coverage(word, instruction_pc, coverage));
                }
            };
            steps = steps.saturating_add(1);
            pc = pc.saturating_add(INSTRUCTION_BYTES);

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
}

impl Error {
    /// Returns the coverage record carried by an execution error, if applicable.
    #[must_use]
    pub const fn coverage(&self) -> Option<InstructionCoverage> {
        match self {
            Self::TextTooLarge { .. }
            | Self::TruncatedInstruction { .. }
            | Self::RegisterFileTooLarge { .. }
            | Self::ZeroStepBudget => None,
            Self::UnsupportedEncoding { coverage, .. }
            | Self::UnsupportedSourceOperand { coverage, .. }
            | Self::RegisterOutOfBounds { coverage, .. }
            | Self::ExecutionBudgetExceeded { coverage, .. }
            | Self::MissingTerminator { coverage, .. } => Some(*coverage),
        }
    }
}

const INSTRUCTION_BYTES: usize = 4;
const VGPR_SOURCE_BASE: u16 = 256;
const VGPR_SOURCE_LIMIT: u16 = VGPR_SOURCE_BASE + 256;
const V_MOV_B32_E32_MASK: u32 = 0xfe01_fe00;
const V_MOV_B32_E32_BASE: u32 = 0x7e00_0200;
const V_ADD_NC_U32_E32_MASK: u32 = 0xfe00_0000;
const V_ADD_NC_U32_E32_BASE: u32 = 0x4a00_0000;
const V_ADD_F32_E32_MASK: u32 = 0xfe00_0000;
const V_ADD_F32_E32_BASE: u32 = 0x0600_0000;
const S_ENDPGM: u32 = 0xbfb0_0000;

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
    EndProgram,
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

fn decode(word: u32) -> core::result::Result<Instruction, DecodeError> {
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

    Err(DecodeError::Encoding)
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
            let result = lane_binary(left, right, |left, right| {
                (f32::from_bits(left) + f32::from_bits(right)).to_bits()
            });
            write_register(registers, destination, result, pc, *coverage)?;
            coverage.add_f32 = coverage.add_f32.saturating_add(1);
        }
        Instruction::EndProgram => {
            coverage.end_program = coverage.end_program.saturating_add(1);
            return Ok(true);
        }
    }
    Ok(false)
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
    reason = "fixtures use expect to make a failed test assertion name its invariant"
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
}
