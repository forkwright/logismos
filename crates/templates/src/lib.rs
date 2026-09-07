//! Format-neutral immutable rendering of one bounded artifact template.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod error;

use std::io::{self, Write};

use minijinja::{Environment, UndefinedBehavior};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::Serialize;
use snafu::ResultExt;

use crate::error::{
    AllocationSnafu, InvalidConfigurationSnafu, LimitExceededSnafu, RenderedUtf8Snafu,
    TemplateSnafu,
};

pub use crate::error::{Error, Result};

/// Static bounds for one template source and each independent render.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TemplateLimits {
    /// Maximum UTF-8 bytes accepted for the template source.
    template_bytes: usize,
    /// Maximum UTF-8 bytes retained for one rendered result.
    rendered_bytes: usize,
    /// `MiniJinja` instruction budget for each render.
    fuel: u64,
    /// `MiniJinja` recursion limit for each render.
    recursion: usize,
}

impl TemplateLimits {
    /// Construct limits after validating their independent renderer invariants.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] when a limit is zero or the
    /// `fuel` value cannot be represented by `MiniJinja`.
    pub fn new(
        template_bytes: usize,
        rendered_bytes: usize,
        fuel: u64,
        recursion: usize,
    ) -> Result<Self> {
        let limits = Self {
            template_bytes,
            rendered_bytes,
            fuel,
            recursion,
        };
        validate_limits(limits)?;
        Ok(limits)
    }
}

/// An immutable borrowed template whose environment has no resolution authority.
pub struct BoundedTemplate<'source> {
    source: &'source str,
    environment: Environment<'source>,
    limits: TemplateLimits,
}

impl<'source> BoundedTemplate<'source> {
    /// Admit and compile one source under immutable, bounded rendering policy.
    ///
    /// # Errors
    ///
    /// Returns a typed limit, configuration, or `MiniJinja` compilation error.
    pub fn new(source: &'source str, limits: TemplateLimits) -> Result<Self> {
        check_limit("template bytes", source.len(), limits.template_bytes)?;
        let environment = configured_environment(limits)?;
        environment
            .template_from_str(source)
            .context(TemplateSnafu)?;
        Ok(Self {
            source,
            environment,
            limits,
        })
    }

    /// Render one serializable context with fresh fuel and bounded output storage.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, limit, UTF-8, or `MiniJinja` rendering error.
    pub fn render(&self, context: impl Serialize) -> Result<String> {
        let template = self
            .environment
            .template_from_str(self.source)
            .context(TemplateSnafu)?;
        let mut output = ByteCappedWriter::new(self.limits.rendered_bytes)?;
        let result = template.render_captured_to(context, &mut output);
        if output.exceeded {
            return LimitExceededSnafu {
                field: "rendered template bytes",
                actual: output.attempted,
                limit: self.limits.rendered_bytes,
            }
            .fail();
        }
        result.context(TemplateSnafu)?;
        output.into_string()
    }
}

fn validate_limits(limits: TemplateLimits) -> Result<()> {
    if [
        limits.template_bytes,
        limits.rendered_bytes,
        limits.recursion,
    ]
    .contains(&0)
        || limits.fuel == 0
        || isize::try_from(limits.fuel).is_err()
    {
        return InvalidConfigurationSnafu {
            rule: "template limits must be non-zero and fuel must fit isize",
        }
        .fail();
    }
    Ok(())
}

fn configured_environment<'source>(limits: TemplateLimits) -> Result<Environment<'source>> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.set_fuel(Some(limits.fuel));
    environment.set_recursion_limit(limits.recursion);
    if environment.recursion_limit() != limits.recursion {
        return InvalidConfigurationSnafu {
            rule: "requested template recursion limit is not supported by this runtime",
        }
        .fail();
    }
    Ok(environment)
}

fn check_limit(field: &'static str, actual: usize, limit: usize) -> Result<()> {
    if actual > limit {
        return LimitExceededSnafu {
            field,
            actual,
            limit,
        }
        .fail();
    }
    Ok(())
}

struct ByteCappedWriter {
    output: Vec<u8>,
    maximum: usize,
    attempted: usize,
    exceeded: bool,
}

impl ByteCappedWriter {
    fn new(maximum: usize) -> Result<Self> {
        let mut output = Vec::new();
        output.try_reserve_exact(maximum).context(AllocationSnafu {
            target: "rendered template bytes",
        })?;
        Ok(Self {
            output,
            maximum,
            attempted: 0,
            exceeded: false,
        })
    }

    fn into_string(self) -> Result<String> {
        String::from_utf8(self.output).context(RenderedUtf8Snafu)
    }
}

impl Write for ByteCappedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let attempted = self.output.len().checked_add(buffer.len());
        self.attempted = match attempted {
            Some(value) => value,
            None => usize::MAX,
        };
        if attempted.is_none_or(|value| value > self.maximum) {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "output limit reached",
            ));
        }
        self.output.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use minijinja::ErrorKind;
    use serde::Serialize;

    use super::{BoundedTemplate, ByteCappedWriter, Error, TemplateLimits};

    type TestResult<T> = std::result::Result<T, Box<dyn StdError>>;

    #[derive(Serialize)]
    struct Message<'content> {
        content: &'content str,
    }

    #[derive(Serialize)]
    struct Context<'content> {
        message: Message<'content>,
    }

    fn limits() -> TemplateLimits {
        TemplateLimits {
            template_bytes: 4_096,
            rendered_bytes: 256,
            fuel: 10_000,
            recursion: 16,
        }
    }

    #[test]
    fn compile_refuses_invalid_static_limits() {
        assert!(matches!(
            TemplateLimits::new(4_096, 256, 0, 16),
            Err(Error::InvalidConfiguration { .. })
        ));

        assert!(matches!(
            TemplateLimits::new(4_096, 256, u64::MAX, 16),
            Err(Error::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn resolution_has_no_registered_or_host_loader_authority() -> TestResult<()> {
        for source in [
            "{% include 'missing' %}",
            "{% set target = 'missing' %}{% include target %}",
            "{% import 'missing' as imported %}",
            "{% set target = 'missing' %}{% import target as imported %}",
            "{% extends 'missing' %}{% block body %}x{% endblock %}",
            "{% set target = 'missing' %}{% extends target %}{% block body %}x{% endblock %}",
            "{% include '<string>' %}",
            "{% import '<string>' as imported %}",
            "{% extends '<string>' %}{% block body %}x{% endblock %}",
            "{% include 'artifact-chat-template' %}",
            "{% import 'artifact-chat-template' as imported %}",
            "{% extends 'artifact-chat-template' %}{% block body %}x{% endblock %}",
        ] {
            let template = BoundedTemplate::new(source, limits())?;
            assert!(
                matches!(template.render(()), Err(Error::Template { source, .. }) if source.kind() == ErrorKind::TemplateNotFound),
                "source unexpectedly resolved: {source}"
            );
        }
        assert_eq!(
            BoundedTemplate::new("a{% include 'missing' ignore missing %}b", limits())?
                .render(())?,
            "ab"
        );
        assert!(matches!(
            BoundedTemplate::new("{{ missing }}", limits())?.render(()),
            Err(Error::Template { source, .. }) if source.kind() == ErrorKind::UndefinedError
        ));
        Ok(())
    }

    #[test]
    fn supports_macros_json_and_python_compatibility() -> TestResult<()> {
        let python = BoundedTemplate::new("{{ message.content.startswith('a') }}", limits())?
            .render(Context {
                message: Message { content: "alice" },
            })?;
        assert_eq!(python, "True");

        let json =
            BoundedTemplate::new("{{ message.content|tojson }}", limits())?.render(Context {
                message: Message { content: "alice" },
            })?;
        assert_eq!(json, "\"alice\"");

        let macro_template = BoundedTemplate::new(
            "{% macro emit(value) %}{{ value }}{% endmacro %}{{ emit(message.content) }}",
            limits(),
        )?;
        assert_eq!(
            macro_template.render(Context {
                message: Message { content: "alice" },
            })?,
            "alice"
        );
        Ok(())
    }

    #[test]
    fn each_render_receives_fresh_fuel() -> TestResult<()> {
        let mut constrained = limits();
        constrained.fuel = 100;
        let template =
            BoundedTemplate::new("{% for value in range(2) %}x{% endfor %}", constrained)?;
        assert_eq!(template.render(())?, "xx");
        assert_eq!(template.render(())?, "xx");
        Ok(())
    }

    #[test]
    fn caps_and_recursion_refuse_exact_boundaries() -> TestResult<()> {
        let mut capped = limits();
        capped.template_bytes = 4;
        assert!(matches!(
            BoundedTemplate::new("hello", capped),
            Err(Error::LimitExceeded {
                field: "template bytes",
                actual: 5,
                limit: 4,
                ..
            })
        ));

        capped = limits();
        capped.rendered_bytes = 4;
        let template = BoundedTemplate::new("12345", capped)?;
        assert!(matches!(
            template.render(()),
            Err(Error::LimitExceeded {
                field: "rendered template bytes",
                actual: 5,
                limit: 4,
                ..
            })
        ));

        capped = limits();
        capped.recursion = usize::MAX;
        assert!(matches!(
            BoundedTemplate::new("hello", capped),
            Err(Error::InvalidConfiguration { .. })
        ));
        Ok(())
    }

    #[test]
    fn fuel_exhaustion_retains_engine_error_kind() -> TestResult<()> {
        let mut constrained = limits();
        constrained.fuel = 1;
        let template = BoundedTemplate::new(
            "{% for value in range(100) %}hello{% endfor %}",
            constrained,
        )?;
        assert!(matches!(
            template.render(()),
            Err(Error::Template { source, .. }) if source.kind() == ErrorKind::OutOfFuel
        ));
        Ok(())
    }

    #[test]
    fn allocation_and_utf8_failures_retain_typed_sources() -> TestResult<()> {
        let allocation_error = BoundedTemplate::new(
            "x",
            TemplateLimits {
                rendered_bytes: usize::MAX,
                ..limits()
            },
        )?
        .render(())
        .err()
        .ok_or("impossibly large bounded output must not allocate")?;
        assert!(
            StdError::source(&allocation_error)
                .is_some_and(<dyn StdError>::is::<std::collections::TryReserveError>),
            "allocation wrapper must retain TryReserveError"
        );

        let mut invalid_utf8 = ByteCappedWriter::new(1)?;
        invalid_utf8.output.push(0xff);
        let utf8_error = invalid_utf8
            .into_string()
            .err()
            .ok_or("invalid byte must fail UTF-8 conversion")?;
        assert!(
            StdError::source(&utf8_error)
                .is_some_and(<dyn StdError>::is::<std::string::FromUtf8Error>),
            "UTF-8 wrapper must retain FromUtf8Error"
        );
        Ok(())
    }
}
