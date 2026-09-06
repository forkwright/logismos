//! Shared validation for logits and decode-policy parameters.

use snafu::ResultExt;

use crate::error::{
    AllLogitsMaskedSnafu, AllocationSnafu, EmptyLogitsSnafu, InvalidParameterSnafu,
    NonFiniteLogitSnafu, Result, TokenIndexOutOfRangeSnafu, TokenOutOfRangeSnafu,
};

pub(crate) fn validate_logits(logits: &[f32]) -> Result<()> {
    if logits.is_empty() {
        return EmptyLogitsSnafu.fail();
    }

    let mut has_candidate = false;
    for (index, logit) in logits.iter().copied().enumerate() {
        if logit.is_nan() {
            return NonFiniteLogitSnafu { index, kind: "NaN" }.fail();
        }
        if logit == f32::INFINITY {
            return NonFiniteLogitSnafu {
                index,
                kind: "positive infinity",
            }
            .fail();
        }
        if logit.is_finite() {
            has_candidate = true;
        }
    }

    if !has_candidate {
        return AllLogitsMaskedSnafu.fail();
    }
    Ok(())
}

pub(crate) fn validate_probability(name: &'static str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return InvalidParameterSnafu {
            name,
            rule: "must be finite and within 0.0..=1.0",
        }
        .fail();
    }
    Ok(())
}

pub(crate) fn validate_positive(name: &'static str, value: f32) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return InvalidParameterSnafu {
            name,
            rule: "must be finite and greater than zero",
        }
        .fail();
    }
    Ok(())
}

pub(crate) fn token_index(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| TokenIndexOutOfRangeSnafu { index }.build())
}

pub(crate) fn validate_token(token_id: u32, vocabulary: usize) -> Result<usize> {
    let index = usize::try_from(token_id).map_err(|_| {
        TokenOutOfRangeSnafu {
            token_id,
            vocabulary,
        }
        .build()
    })?;
    if index >= vocabulary {
        return TokenOutOfRangeSnafu {
            token_id,
            vocabulary,
        }
        .fail();
    }
    Ok(index)
}

pub(crate) fn reserve<T>(values: &mut Vec<T>, target: &'static str, length: usize) -> Result<()> {
    values
        .try_reserve_exact(length)
        .with_context(|_| AllocationSnafu { target, length })
}
