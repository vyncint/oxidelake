//! A Prometheus scrape endpoint for cluster processes (#33, feature `metrics`).
//!
//! A cluster executor's operators report into the process-wide
//! [`TelemetryHub`] — the plan codec attaches it to every `Gpu*Exec` it
//! decodes — and until now nothing could read those counters from outside the
//! process. `oxide-worker --metrics-port 9100` serves them at `/metrics`.
//!
//! ## Why this is hand-written
//!
//! The endpoint answers one path with one fixed content type and reads no
//! request body. An HTTP framework would be several hundred dependencies for
//! a response whose whole grammar is "GET /metrics". The parsing is therefore
//! deliberately narrow: the request line is read up to a bounded length, only
//! `GET /metrics` is answered, and anything else is a 404 or a 400.
//!
//! ## What it is not
//!
//! There is no authentication and no TLS, exactly as the Ballista ports
//! themselves have none (`STATUS.md`, "not addressed"). Bind it on a private
//! interface. It exposes operator names, backends and counts — no query text,
//! no data.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use oxidelake_core::telemetry::TelemetryHub;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

/// The path served. Anything else is a 404.
pub const METRICS_PATH: &str = "/metrics";

/// Longest request line accepted, in bytes. A scraper's is about 20; this is
/// the bound that stops a connection that never sends a newline from growing
/// a buffer without limit.
const MAX_REQUEST_LINE: u64 = 8 * 1024;

/// Serves `hub` as Prometheus text on `addr` until the task is dropped.
///
/// Returns once the listener is bound, so a caller can report the real port
/// (useful when `addr` asks for port 0) before the loop starts.
pub async fn serve(
    addr: SocketAddr,
    hub: Arc<TelemetryHub>,
) -> io::Result<(SocketAddr, impl Future<Output = ()>)> {
    let listener = TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    Ok((local, async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    let hub = Arc::clone(&hub);
                    tokio::spawn(async move {
                        if let Err(e) = handle(stream, &hub).await {
                            tracing::debug!(%peer, error = %e, "metrics request failed");
                        }
                    });
                }
                // A failed accept is not a reason to stop serving metrics:
                // the common causes (fd exhaustion, a peer that vanished
                // between the SYN and the accept) are transient, and a
                // metrics endpoint that silently stops is worse than one that
                // logs and carries on.
                Err(e) => tracing::warn!(error = %e, "metrics listener accept failed"),
            }
        }
    }))
}

async fn handle(stream: TcpStream, hub: &TelemetryHub) -> io::Result<()> {
    let (read, mut write) = stream.into_split();
    let mut line = String::new();
    BufReader::new(read.take(MAX_REQUEST_LINE))
        .read_line(&mut line)
        .await?;
    let mut parts = line.split_whitespace();
    let response = match (parts.next(), parts.next()) {
        (Some("GET"), Some(path)) if path.split('?').next() == Some(METRICS_PATH) => {
            let body = hub.snapshot().to_prometheus();
            format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            )
        }
        (Some("GET"), Some(_)) => status(404, "Not Found", "only /metrics is served\n"),
        (Some(_), Some(_)) => status(405, "Method Not Allowed", "only GET is served\n"),
        _ => status(400, "Bad Request", "expected a request line\n"),
    };
    write.write_all(response.as_bytes()).await?;
    write.flush().await
}

fn status(code: u16, reason: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {code} {reason}\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use oxidelake_core::BackendKind;

    use super::*;

    async fn request(addr: SocketAddr, line: &str) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(line.as_bytes()).await.unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn metrics_are_served_and_other_paths_are_not() {
        let hub = TelemetryHub::new();
        let op = hub.register_operator("GpuFilterExec", BackendKind::Cuda);
        op.record_batch(100, 40, std::time::Duration::from_millis(1));

        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let (bound, server) = serve(addr, Arc::clone(&hub)).await.unwrap();
        let handle = tokio::spawn(server);

        let ok = request(bound, "GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n").await;
        assert!(ok.starts_with("HTTP/1.1 200 OK"), "{ok}");
        assert!(ok.contains("text/plain; version=0.0.4"), "{ok}");
        assert!(
            ok.contains("oxide_operator_rows_in_total{operator=\"GpuFilterExec\""),
            "{ok}"
        );

        // A scraper that appends a query string still gets the metrics.
        let query = request(bound, "GET /metrics?x=1 HTTP/1.1\r\n\r\n").await;
        assert!(query.starts_with("HTTP/1.1 200 OK"), "{query}");

        let missing = request(bound, "GET /health HTTP/1.1\r\n\r\n").await;
        assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");

        let post = request(bound, "POST /metrics HTTP/1.1\r\n\r\n").await;
        assert!(post.starts_with("HTTP/1.1 405"), "{post}");

        handle.abort();
    }

    /// The counters are live: a scrape after more work reports more work.
    #[tokio::test]
    async fn a_second_scrape_sees_new_counters() {
        let hub = TelemetryHub::new();
        let op = hub.register_operator("GpuAggregateExec", BackendKind::Metal);
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let (bound, server) = serve(addr, Arc::clone(&hub)).await.unwrap();
        let handle = tokio::spawn(server);

        let before = request(bound, "GET /metrics HTTP/1.1\r\n\r\n").await;
        assert!(
            before.contains(
                "oxide_operator_batches_total{operator=\"GpuAggregateExec\",backend=\"metal\"} 0"
            ),
            "{before}"
        );
        op.record_batch(10, 2, std::time::Duration::from_millis(1));
        op.record_fallback();
        let after = request(bound, "GET /metrics HTTP/1.1\r\n\r\n").await;
        assert!(
            after.contains(
                "oxide_operator_batches_total{operator=\"GpuAggregateExec\",backend=\"metal\"} 1"
            ),
            "{after}"
        );
        assert!(after.contains("oxide_operator_fallback_batches_total{operator=\"GpuAggregateExec\",backend=\"metal\"} 1"), "{after}");

        handle.abort();
    }
}
