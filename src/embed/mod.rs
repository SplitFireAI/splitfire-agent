//! The SplitFire-only pieces a Tauri host needs beside [`ed-acp-tauri`], which carries the
//! app-neutral half of embedding (the in-process ACP client, sessions, permissions, the key
//! store). These three stay here because they are specific to SplitFire's backend and apps:
//!
//! - [`fetch_key`]: ask the SplitFire API for the signed-in user's Onde key.
//! - [`splitfire_mcp`]: the SplitFire API's MCP server for a session.
//! - [`trim`]: drop `run_command` and swap the stems guidance on iOS.
//!
//! Nothing here depends on `tauri`. Anything that needed an `AppHandle` in the old
//! `splitfire_agent_host` crate takes the app identifier as a `&str` instead, so the host
//! passes `app.config().identifier`.
//!
//! [`ed-acp-tauri`]: https://crates.io/crates/ed-acp-tauri

pub mod trim;

use ed_acp::agent_client_protocol::schema::v1::{HttpHeader, McpServer, McpServerHttp};
use serde::Deserialize;
use tracing::warn;

/// What [`fetch_key`] rejects with when the SplitFire session is missing or
/// expired: the user signs in again.
pub const SIGN_IN_REQUIRED: &str = "sign_in_required";

/// What [`fetch_key`] rejects with when SplitFire can't issue a key now: the
/// service is off, or the account isn't allowed one.
pub const UNAVAILABLE: &str = "unavailable";

/// Name of the SplitFire API's MCP server in agent sessions.
pub const SPLITFIRE_MCP: &str = "splitfire";

/// The agent key SplitFire hands a signed-in user, and the model tier it goes with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentKey {
    /// The Onde key, `app-id:app-secret`. Never logged.
    pub key: String,
    /// The tier SplitFire chose for this app. Older servers don't send it.
    pub model: Option<String>,
}

#[derive(Deserialize)]
struct KeyResponse {
    key: String,
    model: Option<String>,
}

/// Ask SplitFire for the signed-in user's Onde key and the model to use with it. `url` is the
/// app's `/api/v1/agent/key`; `app_identifier` goes out as sent, so a debug build is told
/// apart from the release app and SplitFire picks its model. The key is never logged.
pub async fn fetch_key(
    url: &str,
    access_token: &str,
    app_identifier: &str,
) -> Result<AgentKey, String> {
    let response = reqwest::Client::new()
        .post(url)
        .bearer_auth(access_token)
        .header("X-App-Identifier", app_identifier)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| {
            warn!("Asking SplitFire for an agent key failed: {e}");
            "Couldn't reach SplitFire. Check your connection and try again.".to_string()
        })?;
    let status = response.status();
    match status.as_u16() {
        200 => {
            let body = response
                .bytes()
                .await
                .map_err(|e| format!("Reading SplitFire's answer failed: {e}"))?;
            serde_json::from_slice::<KeyResponse>(&body)
                .map(|parsed| AgentKey {
                    key: parsed.key,
                    model: parsed.model,
                })
                .map_err(|_| "SplitFire answered with something unexpected.".to_string())
        }
        401 => Err(SIGN_IN_REQUIRED.to_string()),
        403 | 503 => Err(UNAVAILABLE.to_string()),
        other => Err(format!(
            "SplitFire answered {other}. Try again in a moment."
        )),
    }
}

/// The SplitFire API's MCP server for a signed-in user, as an ACP HTTP MCP server. `url` is the
/// app's `/api/v1/mcp`. ed-acp connects to it per session; its tools reach the model as
/// `mcp__splitfire__*`. The server answers with the tool list for the app named in
/// `X-App-Identifier` (the app's own identifier, debug builds as sent). Nothing without a token.
pub fn splitfire_mcp(
    url: String,
    access_token: Option<String>,
    app_identifier: &str,
) -> Vec<McpServer> {
    let Some(token) = access_token.filter(|t| !t.is_empty()) else {
        return Vec::new();
    };
    vec![McpServer::Http(
        McpServerHttp::new(SPLITFIRE_MCP, url).headers(vec![
            HttpHeader::new("Authorization", format!("Bearer {token}")),
            HttpHeader::new("X-App-Identifier", app_identifier.to_string()),
        ]),
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve one canned HTTP answer on a free port. Returns the URL and a handle
    /// that yields the request the server saw.
    fn stub(status: &str, body: &str) -> (String, std::thread::JoinHandle<String>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/api/v1/agent/key", listener.local_addr().unwrap());
        let answer = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut seen = vec![0u8; 4096];
            let n = stream.read(&mut seen).unwrap();
            stream.write_all(answer.as_bytes()).unwrap();
            String::from_utf8_lossy(&seen[..n]).to_string()
        });
        (url, handle)
    }

    fn ask(status: &str, body: &str) -> (Result<AgentKey, String>, String) {
        let (url, server) = stub(status, body);
        let result = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(fetch_key(
                &url,
                "user-token",
                "ai.splitfire.KaroKoweJam-Debug",
            ));
        (result, server.join().unwrap())
    }

    #[test]
    fn the_key_comes_from_a_post_with_the_users_token() {
        let (result, request) = ask("200 OK", r#"{"key":"101:secret","model":"onde-balanced"}"#);
        assert_eq!(
            result,
            Ok(AgentKey {
                key: "101:secret".to_string(),
                model: Some("onde-balanced".to_string()),
            })
        );
        assert!(request.starts_with("POST /api/v1/agent/key"), "{request}");
        assert!(
            request
                .to_lowercase()
                .contains("authorization: bearer user-token"),
            "{request}"
        );
        // The app id goes out unchanged, so a debug build is told apart.
        assert!(
            request
                .to_lowercase()
                .contains("x-app-identifier: ai.splitfire.karokowejam-debug"),
            "{request}"
        );
    }

    #[test]
    fn an_older_server_sends_no_model() {
        let (result, _) = ask("200 OK", r#"{"key":"101:secret"}"#);
        assert_eq!(
            result,
            Ok(AgentKey {
                key: "101:secret".to_string(),
                model: None,
            })
        );
    }

    #[test]
    fn a_missing_session_and_an_unavailable_service_have_their_own_answers() {
        assert_eq!(
            ask("401 Unauthorized", "{}").0,
            Err(SIGN_IN_REQUIRED.to_string())
        );
        assert_eq!(ask("403 Forbidden", "{}").0, Err(UNAVAILABLE.to_string()));
        assert_eq!(
            ask("503 Service Unavailable", "{}").0,
            Err(UNAVAILABLE.to_string())
        );
    }

    #[test]
    fn other_failures_say_so_without_leaking_the_body() {
        let error = ask("502 Bad Gateway", r#"{"error":"secret-detail"}"#)
            .0
            .unwrap_err();
        assert!(error.contains("502"), "{error}");
        assert!(!error.contains("secret-detail"), "{error}");
        let error = ask("200 OK", "not json").0.unwrap_err();
        assert!(error.contains("unexpected"), "{error}");
    }

    #[test]
    fn mcp_needs_a_non_empty_token() {
        assert!(splitfire_mcp("https://api/x".to_string(), None, "app").is_empty());
        assert!(splitfire_mcp("https://api/x".to_string(), Some(String::new()), "app").is_empty());
    }

    #[test]
    fn mcp_carries_the_token_and_app_identifier() {
        let servers = splitfire_mcp(
            "https://api.splitfire.ai/api/v1/mcp".to_string(),
            Some("user-token".to_string()),
            "ai.splitfire.KaroKoweJam",
        );
        let McpServer::Http(http) = servers.into_iter().next().expect("one server") else {
            panic!("expected an HTTP MCP server");
        };
        assert_eq!(http.name, SPLITFIRE_MCP);
        assert_eq!(http.url, "https://api.splitfire.ai/api/v1/mcp");
        let header = |name: &str| {
            http.headers
                .iter()
                .find(|h| h.name == name)
                .map(|h| h.value.clone())
        };
        assert_eq!(
            header("Authorization"),
            Some("Bearer user-token".to_string())
        );
        assert_eq!(
            header("X-App-Identifier"),
            Some("ai.splitfire.KaroKoweJam".to_string())
        );
    }
}
