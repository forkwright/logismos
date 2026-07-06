//! Weight-name mapping.
//!
//! The loader reads tensor names verbatim out of the archive. Each
//! model family in `decoders` / `encoders` / `embed` owns a
//! translation table that renames HF-canonical names onto the
//! logismos-internal keys its block implementations expect.
//!
//! [`NameMap`] is a thin ordered lookup: exact-match first, then
//! prefix-rewrite rules in declaration order. It deliberately stops
//! short of regex — a declarative, transparent mapping is easier to
//! audit and always correct for the names produced by stable HF model
//! exports.

use std::collections::HashMap;

/// Single rename rule: rewrite a matching prefix.
#[derive(Debug, Clone)]
pub struct PrefixRule {
    /// Prefix to match against the source name.
    pub from: String,
    /// Prefix to substitute when the rule fires.
    pub to: String,
}

/// Ordered translation table.
#[derive(Debug, Default, Clone)]
pub struct NameMap {
    exact: HashMap<String, String>,
    prefix: Vec<PrefixRule>,
}

impl NameMap {
    /// Fresh empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an exact-match rename.
    pub fn insert_exact(&mut self, src: impl Into<String>, dst: impl Into<String>) -> &mut Self {
        self.exact.insert(src.into(), dst.into());
        self
    }

    /// Add a prefix-rewrite rule. Rules fire in declaration order.
    pub fn insert_prefix(&mut self, from: impl Into<String>, to: impl Into<String>) -> &mut Self {
        self.prefix.push(PrefixRule {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// Apply the map to `src`. Returns the translated name; if no rule
    /// matches, returns `src` verbatim.
    #[must_use]
    pub fn apply<'a>(&self, src: &'a str) -> std::borrow::Cow<'a, str> {
        if let Some(dst) = self.exact.get(src) {
            return std::borrow::Cow::Owned(dst.clone());
        }
        for rule in &self.prefix {
            if let Some(rest) = src.strip_prefix(&rule.from) {
                let mut out = String::with_capacity(rule.to.len() + rest.len());
                out.push_str(&rule.to);
                out.push_str(rest);
                return std::borrow::Cow::Owned(out);
            }
        }
        std::borrow::Cow::Borrowed(src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_wins() {
        let mut m = NameMap::new();
        m.insert_exact("model.norm.weight", "final_norm.weight");
        assert_eq!(m.apply("model.norm.weight"), "final_norm.weight");
        assert_eq!(m.apply("model.norm.bias"), "model.norm.bias");
    }

    #[test]
    fn prefix_rewrites_in_order() {
        let mut m = NameMap::new();
        m.insert_prefix("model.embed_tokens.", "tok_embed.");
        m.insert_prefix("model.layers.0.self_attn.q_proj.", "block.0.attn.q.");
        assert_eq!(m.apply("model.embed_tokens.weight"), "tok_embed.weight");
        assert_eq!(
            m.apply("model.layers.0.self_attn.q_proj.weight"),
            "block.0.attn.q.weight"
        );
        assert_eq!(m.apply("untouched.name"), "untouched.name");
    }
}
