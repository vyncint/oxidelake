//! Where a cluster process may listen (#30).
//!
//! The scheduler's and the workers' ports have no authentication and no
//! encryption: anyone who can reach them can run a query, and a query can
//! read any file the worker process can read. Until OxideLake has TLS and
//! authentication, the private network is the boundary, and this module
//! holds a process to it: a listener on a public address, or on every
//! address at once (`0.0.0.0`, `::`), is refused unless the operator says,
//! with `--allow-insecure-bind`, that something else (a firewall, an
//! authenticating proxy) already decides who can connect.

use std::net::{IpAddr, ToSocketAddrs};

use oxidelake_core::EngineError;

/// The flag that accepts a listener the private network does not enclose.
pub const ALLOW_INSECURE_BIND: &str = "--allow-insecure-bind";

/// What a listening address is exposed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Exposure {
    /// This machine only.
    Loopback,
    /// A private network: RFC 1918, IPv6 unique-local, or link-local.
    Private,
    /// Any network that routes to it, which may be the internet; also every
    /// interface at once (`0.0.0.0`, `::`), public ones included.
    Public,
}

/// What listening on `ip` exposes a port to.
pub fn exposure(ip: IpAddr) -> Exposure {
    // An IPv4-mapped IPv6 address is the IPv4 address it carries.
    let ip = match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        v4 => v4,
    };
    if ip.is_loopback() {
        return Exposure::Loopback;
    }
    let private = match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => v6.is_unique_local() || v6.is_unicast_link_local(),
    };
    if private {
        Exposure::Private
    } else {
        Exposure::Public
    }
}

/// Checks that listening on `host` (given as `flag`) keeps the port off any
/// public network, unless `allow_insecure` (#30). Returns the widest
/// exposure among the addresses `host` names, so the caller can say where
/// it listens.
///
/// A host that resolves to several addresses is judged by the widest, and
/// one that does not resolve is an error: an address that cannot be checked
/// is not one to listen on.
pub fn check_listen_address(
    flag: &str,
    host: &str,
    allow_insecure: bool,
) -> Result<Exposure, EngineError> {
    // `[::1]` is how an IPv6 address is written beside a port, and how the
    // scheduler and Ballista's executor need it, since they parse
    // `host:port`; the brackets are not part of the address.
    let bare = host
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(host);
    let widest = (bare, 0)
        .to_socket_addrs()
        .map_err(|e| EngineError::plan(format!("{flag} {host}: {e}")))?
        .map(|addr| exposure(addr.ip()))
        .max_by_key(|exposure| *exposure as u8)
        .ok_or_else(|| EngineError::plan(format!("{flag} {host} names no address")))?;
    if widest == Exposure::Public && !allow_insecure {
        return Err(EngineError::plan(format!(
            "{flag} {host} would listen where any network that routes to it can connect, and \
             OxideLake's cluster ports have no authentication or encryption: whoever reaches \
             them can run queries that read any file this process can read. Listen on a \
             loopback or private address (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, \
             fc00::/7), or pass {ALLOW_INSECURE_BIND} if a firewall or an authenticating \
             proxy already decides who can connect (SECURITY.md)"
        )));
    }
    Ok(widest)
}

/// Logs, once at startup, that `what` listens beyond this machine (#30).
pub fn warn_unless_loopback(what: &str, host: &str, exposure: Exposure) {
    match exposure {
        Exposure::Loopback => {}
        Exposure::Private => tracing::warn!(
            host,
            "{what} listens on a private network without authentication or encryption: \
             every host that reaches it can run queries"
        ),
        Exposure::Public => tracing::warn!(
            host,
            "{what} listens beyond any private network without authentication or \
             encryption, as {ALLOW_INSECURE_BIND} allows: whatever reaches it can run queries"
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn of(ip: &str) -> Exposure {
        exposure(ip.parse().unwrap())
    }

    #[test]
    fn loopback_and_private_networks_are_told_apart_from_public_ones() {
        for ip in ["127.0.0.1", "127.255.0.9", "::1", "::ffff:127.0.0.1"] {
            assert_eq!(of(ip), Exposure::Loopback, "{ip}");
        }
        for ip in [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.20",
            "169.254.3.4",
            "fd12:3456::1",
            "fc00::1",
            "fe80::1",
            "::ffff:192.168.1.20",
        ] {
            assert_eq!(of(ip), Exposure::Private, "{ip}");
        }
        // The edges of the private ranges, every interface at once, and the
        // shared (carrier) range, which is other customers' network too.
        for ip in [
            "0.0.0.0",
            "::",
            "8.8.8.8",
            "172.15.255.255",
            "172.32.0.0",
            "192.169.0.1",
            "11.0.0.1",
            "100.64.0.1",
            "2001:4860:4860::8888",
            "::ffff:8.8.8.8",
        ] {
            assert_eq!(of(ip), Exposure::Public, "{ip}");
        }
    }

    #[test]
    fn a_public_listener_needs_the_flag_and_a_private_one_does_not() {
        assert_eq!(
            check_listen_address("--bind-host", "127.0.0.1", false).unwrap(),
            Exposure::Loopback
        );
        assert_eq!(
            check_listen_address("--bind-host", "localhost", false).unwrap(),
            Exposure::Loopback
        );
        assert_eq!(
            check_listen_address("--bind-host", "10.1.2.3", false).unwrap(),
            Exposure::Private
        );
        assert_eq!(
            check_listen_address("--bind-host", "[::1]", false).unwrap(),
            Exposure::Loopback
        );
        assert_eq!(
            check_listen_address("--bind-host", "[fd00::7]", false).unwrap(),
            Exposure::Private
        );
        for host in ["0.0.0.0", "::", "[::]", "203.0.113.7"] {
            let err = check_listen_address("--bind-host", host, false)
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("--bind-host {host}")), "{err}");
            assert!(err.contains(ALLOW_INSECURE_BIND), "{err}");
            assert_eq!(
                check_listen_address("--bind-host", host, true).unwrap(),
                Exposure::Public
            );
        }
    }

    #[test]
    fn an_address_that_cannot_be_checked_is_refused() {
        let err = check_listen_address("--bind-host", "no such host .invalid", true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--bind-host no such host .invalid"), "{err}");
    }
}
