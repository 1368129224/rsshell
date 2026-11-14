//! Built-in `ssh` command: an SSH client with OpenSSH config support.
//!
//! Resolves the target host through `~/.ssh/config` (see
//! [`crate::ssh_config`]) and connects via russh:
//!
//! - **direct** — TCP to `host:port`
//! - **ProxyJump** — a `direct-tcpip` tunnel through the jump host
//!   (`user@jump:port` syntax supported)
//! - **ProxyCommand** — the named process is spawned with its stdin/stdout
//!   wired to the SSH transport, exactly like OpenSSH does it; `%h`, `%p`,
//!   `%r`, `%n`, `%d` and `%%` tokens are expanded first. This is what makes
//!   `ncat`, `cloudflared access ssh`, `tailcat ssh` etc. work without any
//!   per-tool adaptation.
//!
//! Authentication tries, in order: the configured identity file (with an
//! optional passphrase prompt), keys held by `ssh-agent`, default identity
//! files, then a password typed interactively. Interactive sessions request
//! a PTY sized to the local terminal and put the terminal into raw mode.

use std::path::PathBuf;
use std::sync::Arc;

use russh::client::{self, Handle};
use russh::keys::PublicKeyOrCertificate;
use russh::keys::agent::client::AgentClient;
use russh::keys::known_hosts::{check_known_hosts, learn_known_hosts};
use russh::keys::{PrivateKey, PrivateKeyWithHashAlg, PublicKey};
use russh::{ChannelMsg, Disconnect, Error as RusshError};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::ssh_config::{SshConfig, SshHostConfig, expand_tokens, load_user_ssh_config};

/// Exit code reported when the SSH connection or authentication fails,
/// mirroring OpenSSH.
pub const SSH_FAILURE: i32 = 255;

/// `-o StrictHostKeyChecking=` behaviour, mirroring OpenSSH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrictHostKeyChecking {
    /// Prompt before recording a new host key (OpenSSH default).
    #[default]
    Ask,
    /// Record new keys automatically, refuse changed keys.
    AcceptNew,
    /// Accept and record any key, allow changed keys (insecure, for scripting).
    No,
    /// Refuse unknown keys; only connect to hosts already known.
    Yes,
}

impl StrictHostKeyChecking {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ask" => Some(Self::Ask),
            "accept-new" | "accept_new" => Some(Self::AcceptNew),
            "no" | "off" => Some(Self::No),
            "yes" | "on" => Some(Self::Yes),
            _ => None,
        }
    }
}

// ───── CLI argument parsing ─────

/// Parsed `ssh` command line.
#[derive(Debug, Default)]
pub struct SshArgs {
    /// Destination: `host`, `user@host`, `ssh://user@host:port/path`,
    /// or a `~/.ssh/config` alias.
    pub destination: String,
    /// `-p port`
    pub port: Option<u16>,
    /// `-l user` (or from `user@host`)
    pub user: Option<String>,
    /// `-i identity_file`
    pub identity_file: Option<PathBuf>,
    /// `-F config_file`
    pub config_file: Option<PathBuf>,
    /// `-o Key=Value` options; ProxyCommand/ProxyJump are honoured, any
    /// other key is kept only to be reported as unsupported.
    pub options: Vec<(String, String)>,
    /// `-o StrictHostKeyChecking=` policy.
    pub strict_host_key_checking: StrictHostKeyChecking,
    /// Remote command; none means an interactive shell.
    pub command: Vec<String>,
}

/// Parse ssh-style arguments. Options are only recognised *before* the
/// destination; everything after the destination is the remote command,
/// exactly like OpenSSH (`ssh host ls -la` passes `-la` through).
pub fn parse_args(args: &[String]) -> Result<SshArgs, String> {
    let mut parsed = SshArgs::default();
    let mut iter = args.iter().peekable();
    let mut positional: Vec<String> = Vec::new();
    while let Some(arg) = iter.next() {
        if arg == "--" {
            positional.extend(iter.cloned());
            break;
        }
        // After the destination, everything is the remote command.
        if !positional.is_empty() || !arg.starts_with('-') {
            positional.push(arg.clone());
            continue;
        }
        match arg.as_str() {
            "-p" => {
                let v = iter.next().ok_or("ssh: option -p requires an argument")?;
                parsed.port = Some(
                    v.parse::<u16>()
                        .map_err(|_| format!("ssh: invalid port: {v}"))?,
                );
            }
            "-l" => {
                let v = iter.next().ok_or("ssh: option -l requires an argument")?;
                parsed.user = Some(v.clone());
            }
            "-i" => {
                let v = iter.next().ok_or("ssh: option -i requires an argument")?;
                parsed.identity_file = Some(crate::ssh_config::expand_tilde_path(v));
            }
            "-F" => {
                let v = iter.next().ok_or("ssh: option -F requires an argument")?;
                parsed.config_file = Some(crate::ssh_config::expand_tilde_path(v));
            }
            "-v" | "-vv" | "-vvv" => { /* verbose logging: accepted, unused */ }
            "-o" => {
                let v = iter.next().ok_or("ssh: option -o requires an argument")?;
                let (key, value) = v
                    .split_once('=')
                    .ok_or_else(|| format!("ssh: invalid -o option (expected Key=Value): {v}"))?;
                apply_o_option(&mut parsed, key.trim(), value.trim())?;
            }
            other => {
                return Err(format!("ssh: unsupported option: {other}"));
            }
        }
    }
    if positional.is_empty() {
        return Err("ssh: destination required".to_owned());
    }
    parsed.destination = positional.remove(0);
    parsed.command = positional;
    Ok(parsed)
}

/// Apply one `-o Key=Value`. The first value wins (OpenSSH semantics):
/// a later `-o User=` never overrides an earlier `-l`, and vice versa.
fn apply_o_option(parsed: &mut SshArgs, key: &str, value: &str) -> Result<(), String> {
    match key.to_ascii_lowercase().as_str() {
        "user" => {
            parsed.user.get_or_insert_with(|| value.to_owned());
        }
        "port" => {
            if parsed.port.is_none() {
                parsed.port = Some(
                    value
                        .parse::<u16>()
                        .map_err(|_| format!("ssh: invalid port: {value}"))?,
                );
            }
        }
        "stricthostkeychecking" => {
            parsed.strict_host_key_checking = StrictHostKeyChecking::from_str(value)
                .ok_or_else(|| format!("ssh: invalid StrictHostKeyChecking value: {value}"))?;
        }
        _ => parsed.options.push((key.to_owned(), value.to_owned())),
    }
    Ok(())
}

/// Split `user@host` / `user@host:port` / `ssh://user@host:port/path` into
/// `(user, host, port)` parts. Port is `None` when not spelled out.
pub fn split_destination(destination: &str) -> (Option<String>, String, Option<u16>) {
    if let Some(url) = destination.strip_prefix("ssh://") {
        let (auth, hostpart) = match url.split_once('@') {
            Some((a, h)) => (Some(a), h),
            None => (None, url),
        };
        let user = auth.filter(|a| !a.is_empty()).map(str::to_owned);
        // Strip any /path suffix, then split off :port when present.
        let hostpart = hostpart.split('/').next().unwrap_or(hostpart);
        let (host, port) = match hostpart.split_once(':') {
            Some((h, p)) => (h, p.parse::<u16>().ok()),
            None => (hostpart, None),
        };
        return (user, host.to_owned(), port);
    }

    let (user, rest) = match destination.split_once('@') {
        Some((u, h)) if !u.is_empty() => (Some(u.to_owned()), h),
        _ => (None, destination),
    };
    let (host, port) = match rest.rsplit_once(':') {
        Some((h, p)) if !h.contains(']') && !p.is_empty() => (h, p.parse::<u16>().ok()),
        _ => (rest, None),
    };
    (user, host.to_owned(), port)
}

// ───── Server host key verification ─────

struct ClientHandler {
    verify: Arc<dyn Fn(&PublicKeyOrCertificate) -> bool + Send + Sync>,
}

impl client::Handler for ClientHandler {
    type Error = RusshError;

    async fn check_server_key(
        &mut self,
        server_public_key: &PublicKeyOrCertificate,
    ) -> Result<bool, Self::Error> {
        Ok((self.verify)(server_public_key))
    }
}

/// Decision about a host key after the known_hosts check.
enum HostKeyDecision {
    /// Key already recorded: accept, no need to write anything.
    Accept,
    /// Unknown key accepted: write it into known_hosts.
    AcceptAndLearn,
    /// Proceed without recording (changed key under `StrictHostKeyChecking=no`,
    /// or a known_hosts read error).
    AcceptOnce,
    /// Refuse the connection.
    Reject,
}

/// Check `~/.ssh/known_hosts`; prompt the user on first contact or
/// mismatch — the same flow OpenSSH uses, honouring `-o StrictHostKeyChecking`.
fn verify_host_key(
    host: &str,
    port: u16,
    key: &PublicKey,
    strict: StrictHostKeyChecking,
) -> HostKeyDecision {
    match check_known_hosts(host, port, key) {
        Ok(true) => HostKeyDecision::Accept,
        Ok(false) => {
            // The key is not in known_hosts: first contact.
            match strict {
                StrictHostKeyChecking::Yes => {
                    eprintln!(
                        "No host key is known for {host} and you have requested strict checking."
                    );
                    HostKeyDecision::Reject
                }
                StrictHostKeyChecking::No | StrictHostKeyChecking::AcceptNew => {
                    HostKeyDecision::AcceptAndLearn
                }
                StrictHostKeyChecking::Ask => {
                    let fp = key.fingerprint(russh::keys::HashAlg::Sha256);
                    print!(
                        "The authenticity of host '{host}' ({port}) can't be established.\n\
                         Key fingerprint is {fp}.\n\
                         Do you want to continue connecting (yes/no)? "
                    );
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                    let mut answer = String::new();
                    if std::io::stdin().read_line(&mut answer).is_err() {
                        return HostKeyDecision::Reject;
                    }
                    match answer.trim().to_lowercase().as_str() {
                        "yes" | "y" => HostKeyDecision::AcceptAndLearn,
                        _ => {
                            eprintln!("Host key verification failed.");
                            HostKeyDecision::Reject
                        }
                    }
                }
            }
        }
        Err(russh::keys::Error::KeyChanged { line }) => {
            if strict == StrictHostKeyChecking::No {
                eprintln!(
                    "Warning: the host key for '{host}' has changed (known_hosts line {line}).\n\
                     Continuing because StrictHostKeyChecking=no."
                );
                HostKeyDecision::AcceptOnce
            } else {
                eprintln!(
                    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\
                     @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n\
                     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\
                     Host key for '{host}' has changed (known_hosts line {line}).\n\
                     This could indicate a man-in-the-middle attack, or that the\n\
                     server was reinstalled. Refusing to connect."
                );
                HostKeyDecision::Reject
            }
        }
        Err(_) => HostKeyDecision::AcceptOnce,
    }
}

/// Verify a server key and record it in `remember` when it is a new
/// first-contact key. Returns whether the connection may proceed.
fn check_and_maybe_remember(
    host: &str,
    port: u16,
    key: &PublicKey,
    strict: StrictHostKeyChecking,
    remember: &std::sync::Mutex<Option<PublicKey>>,
) -> bool {
    match verify_host_key(host, port, key, strict) {
        HostKeyDecision::AcceptAndLearn => {
            if let Ok(mut slot) = remember.lock() {
                *slot = Some(key.clone());
            }
            true
        }
        HostKeyDecision::Accept | HostKeyDecision::AcceptOnce => true,
        HostKeyDecision::Reject => false,
    }
}

/// Build a handler that verifies server keys against known_hosts with the
/// given strictness and stashes an accepted first-contact key for later
/// persistence. Returns the handler together with the slot to read after
/// the handshake.
fn host_key_handler(
    host: &str,
    port: u16,
    strict: StrictHostKeyChecking,
) -> (ClientHandler, Arc<std::sync::Mutex<Option<PublicKey>>>) {
    let remember = Arc::new(std::sync::Mutex::new(None));
    let slot = remember.clone();
    let host = host.to_owned();
    let handler = ClientHandler {
        verify: Arc::new(move |key| match key {
            PublicKeyOrCertificate::PublicKey { key: pk, .. } => {
                check_and_maybe_remember(&host, port, pk, strict, &slot)
            }
            // Certificates: accepted without known_hosts bookkeeping.
            PublicKeyOrCertificate::Certificate(_) => true,
        }),
    };
    (handler, remember)
}

/// Write a first-contact key accepted during the handshake into known_hosts.
fn persist_learned_key(host: &str, port: u16, remember: &std::sync::Mutex<Option<PublicKey>>) {
    if let Ok(slot) = remember.lock()
        && let Some(key) = slot.as_ref()
    {
        let _ = learn_known_hosts(host, port, key);
    }
}

// ───── Transport ─────

/// The byte stream handed to russh. Either a TCP socket, or the stdio of a
/// ProxyCommand child process. Holding the child keeps the tunnel alive;
/// dropping the stream (session over) kills its whole process group (see
/// [`ProxyChild`]).
/// Variants are boxed to keep the enum small (clippy::large_enum_variant).
enum ProxyStream {
    Tcp(tokio::net::TcpStream),
    Proxy(Box<ProxyChild>),
    /// A `direct-tcpip` channel from a jump host, used as a byte stream.
    Channel(Box<russh::ChannelStream<russh::client::Msg>>),
}

/// A ProxyCommand tunnel: SSH bytes flow to the child's stdin and come
/// back from its stdout. On drop the whole process group is killed, so the
/// relay process (a grandchild of the `sh -c` we spawn) cannot leak when
/// the SSH session ends — this is what "the child dies with the connection"
/// requires, and it cannot be done with tokio's `kill_on_drop`, which only
/// signals the direct child.
struct ProxyChild {
    stdin: tokio::process::ChildStdin,
    stdout: tokio::process::ChildStdout,
    child: tokio::process::Child,
}

impl Drop for ProxyChild {
    fn drop(&mut self) {
        if let Some(pid) = self.child.id() {
            // The proxy runs in its own process group (`process_group(0)`),
            // whose PGID equals the direct child's PID. Signalling the
            // group reaches `sh` and every process it spawned.
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGKILL,
            );
            // Reap the direct child so it does not linger as a zombie.
            let _ = self.child.try_wait();
        }
    }
}

impl AsyncRead for ProxyStream {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match self.get_mut() {
            ProxyStream::Tcp(s) => std::pin::Pin::new(s).poll_read(cx, buf),
            // SSH bytes coming back from the proxy process' stdout.
            ProxyStream::Proxy(p) => std::pin::Pin::new(&mut p.stdout).poll_read(cx, buf),
            ProxyStream::Channel(s) => std::pin::Pin::new(s.as_mut()).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for ProxyStream {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        match self.get_mut() {
            ProxyStream::Tcp(s) => std::pin::Pin::new(s).poll_write(cx, buf),
            // SSH bytes we send go to the proxy process' stdin.
            ProxyStream::Proxy(p) => std::pin::Pin::new(&mut p.stdin).poll_write(cx, buf),
            ProxyStream::Channel(s) => std::pin::Pin::new(s.as_mut()).poll_write(cx, buf),
        }
    }
    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        match self.get_mut() {
            ProxyStream::Tcp(s) => std::pin::Pin::new(s).poll_flush(cx),
            ProxyStream::Proxy(p) => std::pin::Pin::new(&mut p.stdin).poll_flush(cx),
            ProxyStream::Channel(s) => std::pin::Pin::new(s.as_mut()).poll_flush(cx),
        }
    }
    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        match self.get_mut() {
            ProxyStream::Tcp(s) => std::pin::Pin::new(s).poll_shutdown(cx),
            ProxyStream::Proxy(p) => std::pin::Pin::new(&mut p.stdin).poll_shutdown(cx),
            ProxyStream::Channel(s) => std::pin::Pin::new(s.as_mut()).poll_shutdown(cx),
        }
    }
}

// ───── Entry point ─────

/// Run the `ssh` builtin with already word-split arguments. Returns a
/// process exit code like external commands do (255 on failure).
pub fn run(args: &[String]) -> i32 {
    let parsed = match parse_args(args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return SSH_FAILURE;
        }
    };

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("ssh: cannot start async runtime: {e}");
            return SSH_FAILURE;
        }
    };
    runtime.block_on(async_run(parsed))
}

/// Fully resolved connection parameters.
struct Resolved {
    host: String,
    port: u16,
    user: String,
    identity: Option<PathBuf>,
    proxy_command: Option<String>,
    proxy_jump: Option<String>,
    strict_host_key_checking: StrictHostKeyChecking,
}

async fn async_run(parsed: SshArgs) -> i32 {
    let config = match &parsed.config_file {
        Some(path) => crate::ssh_config::parse_ssh_config(path),
        None => load_user_ssh_config(),
    };

    // Warn about `-o` options we recognise but do not implement, so the
    // user never believes a silently-ignored setting took effect.
    for (key, _) in &parsed.options {
        if !key.eq_ignore_ascii_case("proxycommand") && !key.eq_ignore_ascii_case("proxyjump") {
            eprintln!("ssh: ignoring unsupported option '{key}'");
        }
    }

    let (url_user, url_host, url_port) = split_destination(&parsed.destination);
    let host_cfg = config.resolve(&url_host, url_port.unwrap_or(22));

    let resolved = Resolved {
        host: host_cfg.hostname.clone(),
        port: parsed.port.unwrap_or(host_cfg.port),
        user: parsed
            .user
            .clone()
            .or(url_user)
            .or_else(|| host_cfg.user.clone())
            .unwrap_or_else(local_user_name),
        identity: parsed
            .identity_file
            .clone()
            .or(host_cfg.identity_file.clone()),
        proxy_command: parsed
            .options
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("proxycommand"))
            .map(|(_, v)| v.clone())
            .or(host_cfg.proxy_command.clone()),
        proxy_jump: parsed
            .options
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("proxyjump"))
            .map(|(_, v)| v.clone())
            .or(host_cfg.proxy_jump.clone()),
        strict_host_key_checking: parsed.strict_host_key_checking,
    };

    // ProxyCommand wins over ProxyJump, as in OpenSSH.
    let stream = if let Some(cmd) = &resolved.proxy_command {
        let cfg = token_config(&resolved, &url_host);
        match spawn_proxy_command(cmd, &cfg, &resolved.user) {
            Ok((stdin, stdout, child)) => ProxyStream::Proxy(Box::new(ProxyChild {
                stdin,
                stdout,
                child,
            })),
            Err(e) => {
                eprintln!("ssh: proxy command failed: {e}");
                return SSH_FAILURE;
            }
        }
    } else if let Some(jump) = &resolved.proxy_jump {
        match connect_via_jump(
            jump,
            &resolved.host,
            resolved.port,
            &config,
            resolved.strict_host_key_checking,
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ssh: {e}");
                return SSH_FAILURE;
            }
        }
    } else {
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tokio::net::TcpStream::connect((resolved.host.as_str(), resolved.port)),
        )
        .await
        {
            Ok(Ok(s)) => ProxyStream::Tcp(s),
            Ok(Err(e)) => {
                eprintln!(
                    "ssh: connect to host {} port {}: {e}",
                    resolved.host, resolved.port
                );
                return SSH_FAILURE;
            }
            Err(_) => {
                eprintln!(
                    "ssh: connect to host {} port {}: timed out",
                    resolved.host, resolved.port
                );
                return SSH_FAILURE;
            }
        }
    };

    connect_and_run(stream, resolved, parsed).await
}

fn local_user_name() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .ok()
        .filter(|u| !u.is_empty())
        .or_else(|| {
            nix::unistd::User::from_uid(nix::unistd::getuid())
                .ok()
                .flatten()
                .map(|u| u.name)
        })
        .unwrap_or_else(|| "root".to_owned())
}

// ───── ProxyCommand ─────

/// The config used for ProxyCommand token expansion: the alias the user
/// typed (`%n`), the *effective* hostname and port (`%h`, `%p`) — so a
/// `-p`/`-o Port=` override is what the proxy command sees, not the
/// config-file port.
fn token_config(resolved: &Resolved, alias: &str) -> SshHostConfig {
    SshHostConfig {
        host: alias.to_owned(),
        hostname: resolved.host.clone(),
        port: resolved.port,
        ..SshHostConfig::default()
    }
}

/// Spawn the ProxyCommand process with its stdio wired as async pipes.
/// Returns `(stdin, stdout, child)`; [`ProxyChild`] keeps the process
/// alive and kills its process group when the tunnel is dropped.
///
/// Follows the OpenSSH specification by executing the expanded command
/// string through `sh -c "<command>"` so shell redirections, variables and
/// pipes function as expected. Spawns into its own process group so
/// foreground terminal signals (like Ctrl-C) are not delivered to the tunnel.
fn spawn_proxy_command(
    template: &str,
    cfg: &SshHostConfig,
    remote_user: &str,
) -> Result<
    (
        tokio::process::ChildStdin,
        tokio::process::ChildStdout,
        tokio::process::Child,
    ),
    String,
> {
    let command_line = expand_tokens(template, cfg, remote_user);
    if command_line.trim().is_empty() {
        return Err("empty ProxyCommand".to_owned());
    }

    let mut command = tokio::process::Command::new("sh");
    command
        .args(["-c", &command_line])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        // Proxy diagnostics flow to our stderr, like OpenSSH.
        .stderr(std::process::Stdio::inherit())
        .process_group(0);

    let mut child = command
        .spawn()
        .map_err(|e| format!("cannot execute proxy command '{command_line}': {e}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "proxy command has no stdin".to_owned())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "proxy command has no stdout".to_owned())?;
    Ok((stdin, stdout, child))
}

// ───── ProxyJump ─────

/// Parse `user@host:port` / `user@host` / `host` jump specs.
pub fn parse_jump_spec(spec: &str) -> (Option<String>, String, u16) {
    let (user, rest) = match spec.split_once('@') {
        Some((u, h)) if !u.is_empty() => (Some(u.to_owned()), h),
        _ => (None, spec),
    };
    let (host, port) = match rest.split_once(':') {
        Some((h, p)) if !p.is_empty() => (h, p.parse::<u16>().unwrap_or(22)),
        _ => (rest, 22),
    };
    (user, host.to_owned(), port)
}

/// Connect to `target:port` through the jump host: an SSH session is
/// opened to the jump host (itself resolved through `~/.ssh/config`, so a
/// jump host behind a ProxyCommand works too) and a `direct-tcpip` channel
/// carries the bytes to the target. The channel is converted to a byte
/// stream and used as the SSH transport for the real connection.
async fn connect_via_jump(
    jump_spec: &str,
    target_host: &str,
    target_port: u16,
    ssh_config: &SshConfig,
    strict: StrictHostKeyChecking,
) -> Result<ProxyStream, String> {
    if jump_spec.contains(',') {
        return Err("multiple jump hosts are not supported yet".to_owned());
    }
    let (jump_user_arg, jump_host_alias, jump_port) = parse_jump_spec(jump_spec);
    let jump_cfg = ssh_config.resolve(&jump_host_alias, jump_port);

    let jump_resolved = Resolved {
        host: jump_cfg.hostname.clone(),
        port: jump_cfg.port,
        user: jump_user_arg
            .or(jump_cfg.user.clone())
            .unwrap_or_else(local_user_name),
        identity: jump_cfg.identity_file.clone(),
        proxy_command: jump_cfg.proxy_command.clone(),
        proxy_jump: None, // no nested jumps for now
        strict_host_key_checking: strict,
    };

    // Transport to the jump host itself (TCP or its own ProxyCommand).
    let jump_stream: ProxyStream = if let Some(cmd) = &jump_resolved.proxy_command {
        let cfg = token_config(&jump_resolved, &jump_host_alias);
        let (stdin, stdout, child) = spawn_proxy_command(cmd, &cfg, &jump_resolved.user)?;
        ProxyStream::Proxy(Box::new(ProxyChild {
            stdin,
            stdout,
            child,
        }))
    } else {
        let s = tokio::net::TcpStream::connect((jump_resolved.host.as_str(), jump_resolved.port))
            .await
            .map_err(|e| {
                format!(
                    "cannot reach jump host {}:{}: {e}",
                    jump_resolved.host, jump_resolved.port
                )
            })?;
        ProxyStream::Tcp(s)
    };

    // Verify the jump host's key exactly like the main connection.
    let (handler, remember) = host_key_handler(&jump_resolved.host, jump_resolved.port, strict);
    let config = Arc::new(client::Config::default());
    let handle = client::connect_stream(config, jump_stream, handler)
        .await
        .map_err(|e| {
            format!(
                "handshake with jump host {}:{} failed: {e}",
                jump_resolved.host, jump_resolved.port
            )
        })?;
    let handle = authenticate(
        handle,
        &jump_resolved.user,
        jump_resolved.identity.as_deref(),
    )
    .await
    .map_err(|e| format!("jump host authentication failed: {e}"))?;

    // Persist a first-contact jump-host key.
    persist_learned_key(&jump_resolved.host, jump_resolved.port, &remember);

    // Ask the jump host to open a TCP channel to the target.
    let channel = handle
        .channel_open_direct_tcpip(target_host, u32::from(target_port), "127.0.0.1", 0)
        .await;
    let channel = match channel {
        Ok(c) => c,
        Err(e) => {
            let _ = handle
                .disconnect(Disconnect::ByApplication, "", "English")
                .await;
            return Err(format!(
                "jump host {} refused tunnel to {target_host}:{target_port}: {e}",
                jump_resolved.host
            ));
        }
    };
    let stream = channel.into_stream();
    Ok(ProxyStream::Channel(Box::new(stream)))
}

// ───── Authentication ─────

async fn authenticate(
    mut handle: Handle<ClientHandler>,
    user: &str,
    identity: Option<&std::path::Path>,
) -> Result<Handle<ClientHandler>, String> {
    // 1. The configured identity file.
    if let Some(path) = identity {
        match try_identity_file(&mut handle, user, path, true).await {
            Ok(true) => return Ok(handle),
            Ok(false) => {}
            Err(e) => eprintln!("ssh: {e}"),
        }
    }

    // 2. Keys held by ssh-agent.
    if let Ok(mut agent) = AgentClient::connect_env().await
        && let Ok(identities) = agent.request_identities().await
    {
        for agent_identity in identities {
            let key = agent_identity.public_key().into_owned();
            let ok = handle
                .authenticate_publickey_with(user, key, None, &mut agent)
                .await
                .is_ok_and(|res| res.success());
            if ok {
                return Ok(handle);
            }
        }
    }

    // 3. Default identity files (skip encrypted ones silently).
    if let Some(home) = dirs::home_dir() {
        for name in ["id_ed25519", "id_ecdsa", "id_rsa"] {
            let path = home.join(".ssh").join(name);
            if path.exists()
                && Some(path.as_path()) != identity
                && try_identity_file(&mut handle, user, &path, false)
                    .await
                    .is_ok_and(|ok| ok)
            {
                return Ok(handle);
            }
        }
    }

    // 4. Interactive password (up to three tries).
    for _ in 0..3 {
        let Some(password) = prompt_password(&format!("{user}@'s password: ")) else {
            break;
        };
        let result = handle
            .authenticate_password(user, &password)
            .await
            .map_err(|e| format!("password authentication error: {e}"))?;
        if result.success() {
            return Ok(handle);
        }
        eprintln!("Permission denied, please try again.");
    }

    Err(format!("all authentication methods failed for {user}"))
}

/// Try one private key file. With `ask_passphrase`, an encrypted key gets a
/// passphrase prompt; otherwise encrypted keys are skipped silently.
async fn try_identity_file(
    handle: &mut Handle<ClientHandler>,
    user: &str,
    path: &std::path::Path,
    ask_passphrase: bool,
) -> Result<bool, String> {
    let key: PrivateKey = match russh::keys::load_secret_key(path, None) {
        Ok(k) => k,
        Err(russh::keys::Error::KeyIsEncrypted) if ask_passphrase => {
            let prompt = format!("Enter passphrase for key '{}': ", path.display());
            let Some(passphrase) = prompt_password(&prompt) else {
                return Ok(false);
            };
            russh::keys::load_secret_key(path, Some(&passphrase))
                .map_err(|e| format!("cannot load key {}: {e}", path.display()))?
        }
        Err(russh::keys::Error::KeyIsEncrypted) => return Ok(false),
        Err(e) => return Err(format!("cannot load key {}: {e}", path.display())),
    };
    let hash = handle
        .best_supported_rsa_hash()
        .await
        .ok()
        .flatten()
        .flatten();
    let result = handle
        .authenticate_publickey(user, PrivateKeyWithHashAlg::new(Arc::new(key), hash))
        .await
        .map_err(|e| format!("publickey authentication error: {e}"))?;
    Ok(result.success())
}

/// RAII guard for raw terminal mode. Disables raw mode automatically on drop,
/// ensuring the user's terminal is never left in a corrupted state even if
/// the session panics, returns early, or encounters an I/O error.
struct RawModeGuard;

impl RawModeGuard {
    fn acquire() -> Result<Self, std::io::Error> {
        crossterm::terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// Read a password from the terminal with echo disabled, handling backspace.
fn prompt_password(prompt: &str) -> Option<String> {
    use crossterm::tty::IsTty;
    if !std::io::stdin().is_tty() {
        return None;
    }
    print!("{prompt}");
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let _guard = RawModeGuard::acquire().ok()?;
    let mut password = String::new();
    let mut stdin = std::io::stdin();
    let mut byte = [0u8; 1];
    loop {
        match std::io::Read::read(&mut stdin, &mut byte) {
            Ok(0) | Err(_) => break,
            Ok(_) => match byte[0] {
                b'\n' | b'\r' => break,
                3 | 4 => {
                    // Ctrl-C or Ctrl-D: abort
                    println!();
                    return None;
                }
                // Backspace (ASCII BS 8 or DEL 127)
                8 | 127 => {
                    password.pop();
                }
                b if !b.is_ascii_control() => {
                    password.push(b as char);
                }
                _ => {}
            },
        }
    }
    drop(_guard);
    println!();
    if password.is_empty() {
        None
    } else {
        Some(password)
    }
}

// ───── Session ─────

async fn connect_and_run(stream: ProxyStream, resolved: Resolved, parsed: SshArgs) -> i32 {
    let (handler, remember_key) = host_key_handler(
        &resolved.host,
        resolved.port,
        resolved.strict_host_key_checking,
    );

    let config = Arc::new(client::Config {
        keepalive_interval: Some(std::time::Duration::from_secs(30)),
        keepalive_max: 3,
        ..client::Config::default()
    });

    let handle = match client::connect_stream(config, stream, handler).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "ssh: handshake with {}:{} failed: {e}",
                resolved.host, resolved.port
            );
            return SSH_FAILURE;
        }
    };

    let authenticated = authenticate(handle, &resolved.user, resolved.identity.as_deref()).await;
    let handle = match authenticated {
        Ok(h) => h,
        Err(e) => {
            eprintln!("ssh: {e}");
            return SSH_FAILURE;
        }
    };

    // Persist an accepted first-contact key into known_hosts.
    persist_learned_key(&resolved.host, resolved.port, &remember_key);

    if parsed.command.is_empty() {
        run_interactive_session(handle).await
    } else {
        run_command_session(handle, &parsed.command).await
    }
}

/// Interactive session: open a PTY, wire local terminal <-> channel.
async fn run_interactive_session(handle: Handle<ClientHandler>) -> i32 {
    use crossterm::tty::IsTty;

    let mut channel = match handle.channel_open_session().await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ssh: cannot open session: {e}");
            return SSH_FAILURE;
        }
    };

    let interactive = std::io::stdin().is_tty();
    let (term, w, h) = if interactive {
        let term = std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".to_owned());
        match crossterm::terminal::size() {
            Ok((w, h)) => (term, u32::from(w), u32::from(h)),
            Err(_) => (term, 80, 24),
        }
    } else {
        ("dumb".to_owned(), 80, 24)
    };

    if let Err(e) = channel.request_pty(false, &term, w, h, 0, 0, &[]).await {
        eprintln!("ssh: pty request failed: {e}");
        return SSH_FAILURE;
    }
    if let Err(e) = channel.request_shell(false).await {
        eprintln!("ssh: shell request failed: {e}");
        return SSH_FAILURE;
    }

    let _raw_guard = if interactive {
        RawModeGuard::acquire().ok()
    } else {
        None
    };

    let mut sigwinch =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::window_change()).ok();

    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut buf = vec![0u8; 4096];
    let mut stdin_open = true;
    let mut exit_code: u32 = 0;

    loop {
        tokio::select! {
            read = stdin.read(&mut buf), if stdin_open => {
                match read {
                    Ok(0) | Err(_) => {
                        stdin_open = false;
                        let _ = channel.eof().await;
                    }
                    Ok(n) => {
                        if channel.data(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                }
            }
            msg = channel.wait() => {
                match msg {
                    Some(ChannelMsg::Data { ref data })
                    | Some(ChannelMsg::ExtendedData { ref data, .. }) => {
                        if stdout.write_all(data).await.is_err() {
                            break;
                        }
                        let _ = stdout.flush().await;
                    }
                    Some(ChannelMsg::ExitStatus { exit_status }) => {
                        exit_code = exit_status;
                    }
                    Some(ChannelMsg::Close) | None => break,
                    _ => {}
                }
            }
            // Propagate terminal window resize (SIGWINCH) to the remote PTY
            _ = async {
                match sigwinch.as_mut() {
                    Some(sig) => sig.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                if let Ok((cols, rows)) = crossterm::terminal::size() {
                    let _ = channel
                        .window_change(u32::from(cols), u32::from(rows), 0, 0)
                        .await;
                }
            }
        }
    }

    drop(_raw_guard);
    println!();
    let _ = handle
        .disconnect(Disconnect::ByApplication, "", "English")
        .await;
    exit_code as i32
}

/// Non-interactive: run one command, stream output, return its exit code.
async fn run_command_session(handle: Handle<ClientHandler>, command: &[String]) -> i32 {
    let command_line = command.join(" ");
    let mut channel = match handle.channel_open_session().await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ssh: cannot open session: {e}");
            return SSH_FAILURE;
        }
    };
    if let Err(e) = channel.exec(false, command_line).await {
        eprintln!("ssh: exec failed: {e}");
        return SSH_FAILURE;
    }

    let mut stdout = tokio::io::stdout();
    let mut exit_code: u32 = 0;
    loop {
        match channel.wait().await {
            Some(ChannelMsg::Data { ref data }) => {
                if stdout.write_all(data).await.is_err() {
                    break;
                }
                let _ = stdout.flush().await;
            }
            Some(ChannelMsg::ExtendedData { ref data, .. }) => {
                let mut stderr = tokio::io::stderr();
                let _ = stderr.write_all(data).await;
                let _ = stderr.flush().await;
            }
            Some(ChannelMsg::ExitStatus { exit_status }) => {
                exit_code = exit_status;
            }
            Some(ChannelMsg::Close) | None => break,
            _ => {}
        }
    }
    let _ = handle
        .disconnect(Disconnect::ByApplication, "", "English")
        .await;
    exit_code as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_basic() {
        // Options come before the destination; `-la` after it is the remote
        // command, as with OpenSSH.
        let args: Vec<String> = ["-p", "2222", "user@host1", "ls", "-la"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let parsed = parse_args(&args).unwrap();
        assert_eq!(parsed.destination, "user@host1");
        assert_eq!(parsed.port, Some(2222));
        assert_eq!(parsed.command, vec!["ls".to_owned(), "-la".to_owned()]);
    }

    #[test]
    fn test_parse_args_options() {
        let args: Vec<String> = [
            "-o",
            "ProxyCommand=ncat --proxy proxy:1080 %h %p",
            "-i",
            "~/.ssh/mykey",
            "host1",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        let parsed = parse_args(&args).unwrap();
        assert_eq!(
            parsed.options[0],
            (
                "ProxyCommand".to_owned(),
                "ncat --proxy proxy:1080 %h %p".to_owned()
            )
        );
        assert_eq!(
            parsed.identity_file,
            Some(crate::ssh_config::expand_tilde_path("~/.ssh/mykey"))
        );
    }

    #[test]
    fn test_parse_args_errors() {
        let empty: Vec<String> = vec![];
        assert!(parse_args(&empty).is_err());
        let bad: Vec<String> = ["-p".to_owned(), "host".to_owned()].to_vec();
        assert!(parse_args(&bad).is_err());
        let badopt: Vec<String> = ["-o".to_owned(), "novalue".to_owned(), "h".to_owned()].to_vec();
        assert!(parse_args(&badopt).is_err());
    }

    #[test]
    fn test_parse_args_o_user_port_strict() {
        let args: Vec<String> = [
            "-o",
            "User=deploy",
            "-o",
            "Port=2222",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "host1",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        let parsed = parse_args(&args).unwrap();
        assert_eq!(parsed.user.as_deref(), Some("deploy"));
        assert_eq!(parsed.port, Some(2222));
        assert_eq!(
            parsed.strict_host_key_checking,
            StrictHostKeyChecking::AcceptNew
        );
        assert!(parsed.options.is_empty());
    }

    #[test]
    fn test_parse_args_o_cli_flag_precedence() {
        // Explicit -l / -p win over -o User / -o Port.
        let args: Vec<String> = [
            "-l",
            "cli-user",
            "-o",
            "User=o-user",
            "-p",
            "2200",
            "-o",
            "Port=2299",
            "host1",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        let parsed = parse_args(&args).unwrap();
        assert_eq!(parsed.user.as_deref(), Some("cli-user"));
        assert_eq!(parsed.port, Some(2200));
    }

    #[test]
    fn test_parse_args_invalid_strict_value() {
        let args: Vec<String> = ["-o", "StrictHostKeyChecking=bogus", "h"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_strict_host_key_checking_from_str() {
        assert_eq!(
            StrictHostKeyChecking::from_str("ask"),
            Some(StrictHostKeyChecking::Ask)
        );
        assert_eq!(
            StrictHostKeyChecking::from_str("ACCEPT-NEW"),
            Some(StrictHostKeyChecking::AcceptNew)
        );
        assert_eq!(
            StrictHostKeyChecking::from_str("no"),
            Some(StrictHostKeyChecking::No)
        );
        assert_eq!(
            StrictHostKeyChecking::from_str("yes"),
            Some(StrictHostKeyChecking::Yes)
        );
        assert_eq!(StrictHostKeyChecking::from_str("bogus"), None);
    }

    #[test]
    fn test_token_config_respects_port_override() {
        // Regression: %p must expand to the *effective* port (a `-p`/`-o
        // Port=` override), not the config-file port the resolver saw.
        let resolved = Resolved {
            host: "real.example.com".to_owned(),
            port: 2222, // overridden on the command line
            user: "admin".to_owned(),
            identity: None,
            proxy_command: None,
            proxy_jump: None,
            strict_host_key_checking: StrictHostKeyChecking::Ask,
        };
        let cfg = token_config(&resolved, "alias");
        assert_eq!(
            expand_tokens("nc %h %p %n", &cfg, "admin"),
            "nc real.example.com 2222 alias"
        );
    }

    #[test]
    fn test_split_destination() {
        assert_eq!(split_destination("host1"), (None, "host1".to_owned(), None));
        assert_eq!(
            split_destination("admin@host1"),
            (Some("admin".to_owned()), "host1".to_owned(), None)
        );
        assert_eq!(
            split_destination("admin@host1:2222"),
            (Some("admin".to_owned()), "host1".to_owned(), Some(2222))
        );
        assert_eq!(
            split_destination("ssh://host1:2222"),
            (None, "host1".to_owned(), Some(2222))
        );
        assert_eq!(
            split_destination("ssh://root@host1"),
            (Some("root".to_owned()), "host1".to_owned(), None)
        );
    }

    #[test]
    fn test_parse_jump_spec() {
        assert_eq!(parse_jump_spec("bastion"), (None, "bastion".to_owned(), 22));
        assert_eq!(
            parse_jump_spec("admin@bastion:2222"),
            (Some("admin".to_owned()), "bastion".to_owned(), 2222)
        );
        assert_eq!(
            parse_jump_spec("bastion:2222"),
            (None, "bastion".to_owned(), 2222)
        );
    }

    #[tokio::test]
    async fn test_proxy_stream_roundtrip_over_tcp() {
        // ProxyStream over TCP behaves like the socket itself: spin up a
        // tiny echo server and exercise both AsyncRead and AsyncWrite.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 5];
            sock.read_exact(&mut buf).await.unwrap();
            sock.write_all(&buf).await.unwrap();
        });
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut stream = ProxyStream::Tcp(tcp);
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        stream.write_all(b"hello").await.unwrap();
        stream.flush().await.unwrap();
        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"hello");
        server.await.unwrap();
    }
}
