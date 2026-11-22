//! Switching the agent between Onde Cloud and on-device inference, for hosts with a settings
//! switch. The provider is read from `<PREFIX>_PROVIDER` in the agent's stored `env` file, which
//! the agent re-reads on every `session/new` and `session/load`, so a host only has to write it
//! and open a session.
//!
//! Switching back to Cloud needs no helper here: the host asks SplitFire for a key again
//! ([`super::fetch_key`]) and stores it with `ed_acp_tauri::set_api_key`, which writes
//! `<PREFIX>_PROVIDER=onde` and the model tier SplitFire chose.

use ed_acp::cli::with_vars;
use ed_acp::{LlmConfig, LlmEnv, Provider};

/// Whether the agent will run on the device: the provider in the environment or stored `env`
/// file is [`Provider::Local`].
pub fn on_device(env: &LlmEnv) -> bool {
    LlmConfig::from_env(env).provider == Provider::Local
}

/// Run the agent on the device from now on. Writes `<PREFIX>_PROVIDER=local` to the stored `env`
/// file (mode 0600) and drops `<PREFIX>_MODEL`, so the per-platform on-device default applies
/// and a Cloud tier name never reaches the local engine. A stored `ONDE_API_KEY` stays, so
/// switching back to Cloud doesn't need a new key.
///
/// The agent has to be built with the `local` feature, or its turns fail saying so.
pub fn use_on_device(env: &LlmEnv) -> Result<(), String> {
    let dir = env.config_dir().ok_or("No config directory available.")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("env");
    let existing = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let model_var = env.var("MODEL");
    let kept: String = existing
        .lines()
        .filter(|line| {
            let line = line.trim();
            let line = line.strip_prefix("export ").unwrap_or(line);
            !line.starts_with(&format!("{model_var}="))
        })
        .map(|line| format!("{line}\n"))
        .collect();
    let content = with_vars(&kept, &[(env.var("PROVIDER"), "local".to_string())]);
    std::fs::write(&path, content).map_err(|e| format!("{}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    tracing::info!("The agent now runs on the device ({})", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway data dir for one test, named so tests can't collide.
    struct TempEnv {
        env: LlmEnv,
        dir: std::path::PathBuf,
    }

    impl TempEnv {
        fn new(prefix: &'static str) -> Self {
            let dir = std::env::temp_dir().join(format!("splitfire-provider-{prefix}"));
            let _ = std::fs::remove_dir_all(&dir);
            // SAFETY: each test uses its own prefix, so no other thread reads this variable.
            unsafe { std::env::set_var(format!("{prefix}_DATA_DIR"), &dir) };
            Self {
                env: LlmEnv {
                    prefix,
                    dir_name: "splitfire-agent",
                    default_model: "onde-kkk",
                },
                dir,
            }
        }
    }

    impl Drop for TempEnv {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn switching_keeps_the_key_and_drops_the_cloud_model() {
        let t = TempEnv::new("SFPROVA");
        std::fs::create_dir_all(&t.dir).unwrap();
        std::fs::write(
            t.dir.join("env"),
            "SFPROVA_PROVIDER=onde\nONDE_API_KEY=app:secret\nSFPROVA_MODEL=onde-balanced\n",
        )
        .unwrap();
        assert!(!on_device(&t.env));

        use_on_device(&t.env).unwrap();

        let written = std::fs::read_to_string(t.dir.join("env")).unwrap();
        assert!(written.contains("ONDE_API_KEY=app:secret"), "{written}");
        assert!(written.contains("SFPROVA_PROVIDER=local"), "{written}");
        assert!(!written.contains("onde-balanced"), "{written}");
        assert!(!written.contains("SFPROVA_PROVIDER=onde"), "{written}");
        assert!(on_device(&t.env));
        // The stored key is not this provider's key.
        assert_eq!(LlmConfig::from_env(&t.env).api_key, None);
    }

    #[test]
    fn switching_works_without_an_env_file() {
        let t = TempEnv::new("SFPROVB");
        use_on_device(&t.env).unwrap();
        assert!(on_device(&t.env));
    }
}
