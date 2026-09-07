//! Typed failures for bounded template rendering.

use snafu::Snafu;

/// Result alias used throughout `templates`.
pub type Result<T> = std::result::Result<T, Error>;

/// Failures from compiling or rendering one bounded template.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// A bounded render buffer allocation could not be reserved.
    #[snafu(display("template renderer could not reserve bounded {target} storage"))]
    Allocation {
        /// Allocation purpose.
        target: &'static str,
        /// Allocation failure returned by the standard library.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Template compilation or rendering failed.
    #[snafu(display("template renderer failed: {source}"))]
    Template {
        /// Template engine failure retaining its error chain.
        source: minijinja::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Rendered bytes violated the UTF-8 invariant required by template output.
    #[snafu(display("template renderer emitted invalid UTF-8: {source}"))]
    RenderedUtf8 {
        /// UTF-8 conversion failure retaining its error chain.
        source: std::string::FromUtf8Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A template source or rendered result exceeded a declared bound.
    #[snafu(display("template renderer {field} {actual} exceeds limit {limit}"))]
    LimitExceeded {
        /// Bounded dimension.
        field: &'static str,
        /// Observed value.
        actual: usize,
        /// Maximum accepted value.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Static renderer configuration was internally inconsistent.
    #[snafu(display("template renderer configuration violates {rule}"))]
    InvalidConfiguration {
        /// Exact invariant that failed.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
