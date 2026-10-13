//! What a host takes out of the agent on a platform that can't start processes,
//! and the stems paragraph an app swaps in when it has its own answer.
//!
//! iOS apps can't start other programs, so `run_command` would only ever fail
//! there and the stock "install demucs" advice is no use. Each app's iOS
//! profile wraps its desktop one with these.

use std::sync::Arc;

use async_trait::async_trait;
use ed_acp::agent_client_protocol::schema::v1::{ToolCallLocation, ToolKind};
use ed_acp::{DescribeCtx, ToolCtx, ToolOutcome, Toolset};
use serde_json::Value;

// The markers the "no stems" paragraph starts and ends with live in `prompt.rs`, which writes
// the paragraph, so the prompt and the replacement below can't drift apart.
pub use crate::prompt::{NO_STEMS_END, NO_STEMS_START};

/// The workspace tool that starts a process.
pub const RUN_COMMAND: &str = "run_command";

/// Swap the stock "no stems" paragraph for `note`; if the stock prompt is
/// worded differently, append the note rather than lose it.
pub fn replace_no_stems(prompt: String, note: &str) -> String {
    if let Some(start) = prompt.find(NO_STEMS_START)
        && let Some(rel) = prompt[start..].find(NO_STEMS_END)
    {
        let end = start + rel + NO_STEMS_END.len();
        return format!("{}{}{}", &prompt[..start], note, &prompt[end..]);
    }
    format!("{prompt}\n\n{note}")
}

/// The name of a tool definition, as the model is offered it.
pub fn tool_name(definition: &Value) -> Option<&str> {
    definition.pointer("/function/name").and_then(Value::as_str)
}

/// `toolsets` with `run_command` taken out of what the model is offered.
pub fn without_commands(toolsets: Vec<Arc<dyn Toolset>>) -> Vec<Arc<dyn Toolset>> {
    toolsets
        .into_iter()
        .map(|inner| Arc::new(NoCommands(inner)) as Arc<dyn Toolset>)
        .collect()
}

struct NoCommands(Arc<dyn Toolset>);

#[async_trait]
impl Toolset for NoCommands {
    fn definitions(&self) -> Vec<Value> {
        self.0
            .definitions()
            .into_iter()
            .filter(|d| tool_name(d) != Some(RUN_COMMAND))
            .collect()
    }

    fn handles(&self, name: &str) -> bool {
        name != RUN_COMMAND && self.0.handles(name)
    }

    fn describe(
        &self,
        name: &str,
        args: &Value,
        ctx: &DescribeCtx<'_>,
    ) -> (String, ToolKind, Vec<ToolCallLocation>) {
        self.0.describe(name, args, ctx)
    }

    async fn execute(
        &self,
        ctx: &ToolCtx,
        tool_call_id: &str,
        name: &str,
        args: Value,
    ) -> anyhow::Result<ToolOutcome> {
        self.0.execute(ctx, tool_call_id, name, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_the_stock_paragraph() {
        let stock =
            format!("before\n{NO_STEMS_START} tell them to install demucs, {NO_STEMS_END}\nafter");
        let out = replace_no_stems(stock, "NOTE");
        assert_eq!(out, "before\nNOTE\nafter");
    }

    #[test]
    fn appends_when_the_wording_changed() {
        let out = replace_no_stems("something else".to_string(), "NOTE");
        assert_eq!(out, "something else\n\nNOTE");
    }

    #[test]
    fn the_markers_bound_the_real_prompt_paragraph() {
        // The prompt the agent ships must still contain the paragraph these markers point at,
        // or a host's `replace_no_stems` would silently append instead of replacing.
        let prompt = crate::prompt::system_prompt("acp", std::path::Path::new("/tmp"), &[], false);
        let start = prompt.find(NO_STEMS_START).expect("start marker present");
        assert!(prompt[start..].contains(NO_STEMS_END), "end marker present");
    }
}
