//! Pure parsing for the configured GPU ISA target and runtime ISA spellings.
//!
//! [`configured_target_token`] is the shared reader used by runtime crates for
//! `contracts/gpu-target.txt`; build scripts may still include that file
//! directly when constructing compiler flags. [`TargetIsa`] validates a base
//! architecture plus optional HIP feature-state suffixes without linking HIP or
//! probing hardware. A syntactically valid suffix is reported as data only; it
//! is not evidence that a compiler, device, or kernel supports the named feature.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::BTreeSet;
use std::fmt;

use snafu::Snafu;

/// The complete configured target token, trimmed of its file terminator.
///
/// Consumers that make decisions from this value should parse it with
/// [`TargetIsa::parse`] or use [`matches_configured_architecture`].
#[must_use]
pub fn configured_target_token() -> &'static str {
    include_str!("../../../contracts/gpu-target.txt").trim()
}

/// One syntactically validated AMD GPU ISA token.
///
/// The base architecture uses `gfx` followed by lowercase ASCII alphanumerics,
/// beginning with a digit. Optional colon-separated features use a lowercase
/// ASCII name followed by `+` or `-`, for example `gfx1100:xnack-`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetIsa<'a> {
    raw: &'a str,
    architecture: &'a str,
    features: Vec<IsaFeature<'a>>,
}

impl<'a> TargetIsa<'a> {
    /// Parse and validate a complete ISA token.
    ///
    /// Validation is syntactic. In particular, an accepted feature name does
    /// not establish that the configured binary semantically supports it.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] for an invalid architecture, malformed feature,
    /// or repeated feature name.
    pub fn parse(raw: &'a str) -> Result<Self, ParseError> {
        let (architecture, suffixes) = raw
            .split_once(':')
            .map_or((raw, None), |(base, suffixes)| (base, Some(suffixes)));
        if !valid_architecture(architecture) {
            return Err(ParseSnafu {
                kind: ParseErrorKind::InvalidArchitecture,
                feature_index: 0_usize,
            }
            .build());
        }

        let mut features = Vec::new();
        let mut names = BTreeSet::new();
        if let Some(suffixes) = suffixes {
            for (index, encoded) in suffixes.split(':').enumerate() {
                let feature = parse_feature(encoded, index)?;
                if !names.insert(feature.name) {
                    return Err(ParseSnafu {
                        kind: ParseErrorKind::DuplicateFeature,
                        feature_index: index,
                    }
                    .build());
                }
                features.push(feature);
            }
        }

        Ok(Self {
            raw,
            architecture,
            features,
        })
    }

    /// Return the complete validated token.
    #[must_use]
    pub const fn as_str(&self) -> &'a str {
        self.raw
    }

    /// Return the base architecture without feature suffixes.
    #[must_use]
    pub const fn architecture(&self) -> &'a str {
        self.architecture
    }

    /// Return validated feature-state suffixes in their declared order.
    #[must_use]
    pub fn features(&self) -> &[IsaFeature<'a>] {
        &self.features
    }
}

/// One syntactically validated ISA feature-state suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsaFeature<'a> {
    name: &'a str,
    state: FeatureState,
}

impl<'a> IsaFeature<'a> {
    /// Return the feature name without its `+` or `-` state marker.
    #[must_use]
    pub const fn name(self) -> &'a str {
        self.name
    }

    /// Return the declared feature state.
    #[must_use]
    pub const fn state(self) -> FeatureState {
        self.state
    }
}

/// Syntactic state carried by one ISA feature suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeatureState {
    /// The token ends in `+`.
    Enabled,
    /// The token ends in `-`.
    Disabled,
}

/// Classification of a target-token parse failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseErrorKind {
    /// The base was not a canonical AMD `gfx...` architecture.
    InvalidArchitecture,
    /// A colon introduced an empty feature segment.
    EmptyFeature,
    /// A feature did not end in `+` or `-`.
    MissingFeatureState,
    /// A feature name was not canonical lowercase ASCII.
    InvalidFeatureName,
    /// The same feature name appeared more than once.
    DuplicateFeature,
}

/// Error returned when an ISA target token is not canonical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Snafu)]
#[snafu(display("invalid GPU ISA token: {kind} at feature index {feature_index}"))]
pub struct ParseError {
    kind: ParseErrorKind,
    feature_index: usize,
    #[snafu(implicit)]
    location: snafu::Location,
}

impl ParseError {
    /// Return the parse-failure classification.
    #[must_use]
    pub const fn kind(self) -> ParseErrorKind {
        self.kind
    }

    /// Return the zero-based feature index associated with the failure.
    ///
    /// Architecture failures report zero because they occur before features.
    #[must_use]
    pub const fn feature_index(self) -> usize {
        self.feature_index
    }
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::InvalidArchitecture => "invalid architecture",
            Self::EmptyFeature => "empty feature",
            Self::MissingFeatureState => "missing feature state",
            Self::InvalidFeatureName => "invalid feature name",
            Self::DuplicateFeature => "duplicate feature",
        };
        formatter.write_str(description)
    }
}

/// Whether `candidate` has the configured base architecture and a valid token.
///
/// Feature suffixes are syntax-checked, including uniqueness, but are not
/// interpreted as a semantic compatibility claim. A malformed configured
/// contract or candidate fails closed.
#[must_use]
pub fn matches_configured_architecture(candidate: &str) -> bool {
    let Ok(configured) = TargetIsa::parse(configured_target_token()) else {
        return false;
    };
    let Ok(candidate) = TargetIsa::parse(candidate) else {
        return false;
    };
    candidate.architecture() == configured.architecture()
}

fn valid_architecture(architecture: &str) -> bool {
    let Some(encoded) = architecture.strip_prefix("gfx") else {
        return false;
    };
    let mut bytes = encoded.bytes();
    matches!(bytes.next(), Some(first) if first.is_ascii_digit())
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
}

fn parse_feature(encoded: &str, index: usize) -> Result<IsaFeature<'_>, ParseError> {
    if encoded.is_empty() {
        return Err(ParseSnafu {
            kind: ParseErrorKind::EmptyFeature,
            feature_index: index,
        }
        .build());
    }
    let (name, state) = if let Some(name) = encoded.strip_suffix('+') {
        (name, FeatureState::Enabled)
    } else if let Some(name) = encoded.strip_suffix('-') {
        (name, FeatureState::Disabled)
    } else {
        return Err(ParseSnafu {
            kind: ParseErrorKind::MissingFeatureState,
            feature_index: index,
        }
        .build());
    };
    let mut bytes = name.bytes();
    if !matches!(bytes.next(), Some(first) if first.is_ascii_lowercase())
        || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(ParseSnafu {
            kind: ParseErrorKind::InvalidFeatureName,
            feature_index: index,
        }
        .build());
    }
    Ok(IsaFeature { name, state })
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions use expect() directly")]

    use super::*;

    #[test]
    fn parses_base_and_feature_states_without_claiming_semantics() {
        let parsed =
            TargetIsa::parse("gfx1100:sramecc+:xnack-").expect("canonical target token must parse");
        assert_eq!(parsed.architecture(), "gfx1100");
        assert_eq!(parsed.as_str(), "gfx1100:sramecc+:xnack-");
        assert_eq!(
            parsed.features(),
            &[
                IsaFeature {
                    name: "sramecc",
                    state: FeatureState::Enabled,
                },
                IsaFeature {
                    name: "xnack",
                    state: FeatureState::Disabled,
                },
            ]
        );
    }

    #[test]
    fn configured_match_rejects_disagreement_malformed_and_duplicate_features() {
        assert!(matches_configured_architecture(configured_target_token()));
        assert!(matches_configured_architecture(&format!(
            "{}:xnack-",
            configured_target_token()
        )));
        for invalid in [
            "gfx1101",
            "gfx11000",
            "gfx1100:xnack",
            "gfx1100::xnack+",
            "gfx1100:XNACK+",
            "gfx1100:xnack+:xnack-",
        ] {
            assert!(
                !matches_configured_architecture(invalid),
                "invalid or disagreeing target matched: {invalid}"
            );
        }
    }

    #[test]
    fn duplicate_feature_has_a_typed_failure() {
        let error = TargetIsa::parse("gfx1100:xnack+:xnack-")
            .expect_err("duplicate feature names must fail");
        assert_eq!(error.kind(), ParseErrorKind::DuplicateFeature);
        assert_eq!(error.feature_index(), 1);
        assert!(
            error.location.file().ends_with("crates/isa/src/lib.rs"),
            "SNAFU must retain the parse failure's source location"
        );
    }
}
