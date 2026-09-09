//! Shared checked GGUF matrix access for native Qwen-family CPU paths.

use snafu::ResultExt;

use loader::gguf::{GgmlType, VerifiedArtifact, VerifiedTensor};
use quant::{RowFormat, row_byte_len, row_decode_f32, row_dot_f32};

use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, PayloadTensorSnafu, ProjectionAllocationSnafu, ProjectionBytesSnafu,
    ProjectionDtypeSnafu, ProjectionInputWidthSnafu, ProjectionLayoutSnafu, ProjectionRankSnafu,
    ProjectionRowSnafu,
};

pub(crate) struct CheckedMatrix<'artifact> {
    name: String,
    tensor: VerifiedTensor<'artifact>,
    format: RowFormat,
    input_width: usize,
    output_width: usize,
    row_byte_len: usize,
}

impl<'artifact> CheckedMatrix<'artifact> {
    pub(crate) const fn projection_output_elements(output_width: usize) -> usize {
        output_width
    }

    pub(crate) const fn decoded_row_elements(input_width: usize) -> usize {
        input_width
    }

    pub(crate) fn from_payload(payload: &'artifact VerifiedArtifact, name: &str) -> Result<Self> {
        let tensor = payload.tensor(name).context(PayloadTensorSnafu {
            name: name.to_string(),
        })?;
        let tensor_name = tensor.name().to_string();
        let [input_width, output_width] = tensor.dims() else {
            return ProjectionRankSnafu {
                name: tensor_name,
                actual: tensor.dims().len(),
            }
            .fail();
        };
        let format = row_format(tensor.ggml_type()).ok_or_else(|| {
            ProjectionDtypeSnafu {
                name: tensor_name.clone(),
                actual: tensor.ggml_type(),
            }
            .build()
        })?;
        let input_width = usize::try_from(*input_width).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "row projection input width",
            }
            .build()
        })?;
        let output_width = usize::try_from(*output_width).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "row projection output width",
            }
            .build()
        })?;
        let row_byte_len =
            row_byte_len(format, input_width).with_context(|_| ProjectionLayoutSnafu {
                name: tensor_name.clone(),
            })?;
        let expected = output_width.checked_mul(row_byte_len).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "row projection payload byte length",
            }
            .build()
        })?;
        let actual = tensor.bytes().len();
        if actual != expected {
            return ProjectionBytesSnafu {
                name: tensor_name,
                expected,
                actual,
            }
            .fail();
        }
        Ok(Self {
            name: tensor.name().to_string(),
            tensor,
            format,
            input_width,
            output_width,
            row_byte_len,
        })
    }

    pub(crate) fn project(&self, activations: &[f32]) -> Result<Vec<f32>> {
        if activations.len() != self.input_width {
            return ProjectionInputWidthSnafu {
                name: self.name.clone(),
                expected: self.input_width,
                actual: activations.len(),
            }
            .fail();
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(Self::projection_output_elements(self.output_width))
            .with_context(|_| ProjectionAllocationSnafu {
                name: self.name.clone(),
                output_width: Self::projection_output_elements(self.output_width),
            })?;
        for (row, row_bytes) in self
            .tensor
            .bytes()
            .chunks_exact(self.row_byte_len)
            .enumerate()
        {
            let value = row_dot_f32(self.format, row_bytes, activations).with_context(|_| {
                ProjectionRowSnafu {
                    name: self.name.clone(),
                    row,
                }
            })?;
            output.push(value);
        }
        Ok(output)
    }

    pub(crate) fn decode_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.output_width {
            return ProjectionInputWidthSnafu {
                name: self.name.clone(),
                expected: self.output_width,
                actual: row,
            }
            .fail();
        }
        let start = row.checked_mul(self.row_byte_len).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "row projection row offset",
            }
            .build()
        })?;
        let end = start.checked_add(self.row_byte_len).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "row projection row end",
            }
            .build()
        })?;
        let row_bytes = self.tensor.bytes().get(start..end).ok_or_else(|| {
            ProjectionBytesSnafu {
                name: self.name.clone(),
                expected: end,
                actual: self.tensor.bytes().len(),
            }
            .build()
        })?;
        row_decode_f32(
            self.format,
            row_bytes,
            Self::decoded_row_elements(self.input_width),
        )
        .with_context(|_| ProjectionRowSnafu {
            name: self.name.clone(),
            row,
        })
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn native_shape(&self) -> std::result::Result<kernels::RowGemvShape, kernels::Error> {
        kernels::RowGemvShape::new(
            self.format,
            self.output_width,
            self.input_width,
            self.tensor.bytes().len(),
            self.input_width,
            self.output_width,
        )
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn serialized_bytes(&self) -> &[u8] {
        self.tensor.bytes()
    }
}

fn row_format(ggml_type: GgmlType) -> Option<RowFormat> {
    match ggml_type {
        GgmlType::F32 => Some(RowFormat::F32),
        GgmlType::Q8_0 => Some(RowFormat::Q8_0),
        GgmlType::Q4K => Some(RowFormat::Q4K),
        GgmlType::Q5K => Some(RowFormat::Q5K),
        GgmlType::Q6K => Some(RowFormat::Q6K),
        GgmlType::IQ4NL => Some(RowFormat::IQ4NL),
        GgmlType::IQ4XS => Some(RowFormat::IQ4XS),
        _ => None,
    }
}
