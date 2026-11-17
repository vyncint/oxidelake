//! Corrupt Parquet comes back as `EngineError::Format`: never a panic, an
//! abort, or an allocation the size of whatever the file declares.
//!
//! Every case runs in a child process under an address-space cap. The
//! failure guarded against includes allocations sized by a corrupt length
//! field: uncapped, a regression would try to take the machine's memory
//! instead of failing the test. Under the cap the allocation fails, the
//! child aborts, and the parent reports which case did it. RLIMIT_AS is only
//! enforced on Linux, so the file is Linux-only rather than uncapped
//! anywhere.

#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use oxidelake_core::EngineError;
use oxidelake_storage::{
    Compression, ParquetScanGuard, ParquetWriteOptions, classify_error, default_object_store,
    pruning_session, register_local_store, register_parquet_table, with_pruning, write_parquet,
};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::{
    ColumnChunkMetaDataBuilder, PageIndexPolicy, ParquetMetaData, ParquetMetaDataBuilder,
    ParquetMetaDataReader, ParquetMetaDataWriter,
};
use parquet::file::page_index::offset_index::PageLocation;
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::file::writer::TrackedWrite;

/// Prefix of every line the child reports, so the parent can find them in
/// the test harness's own output.
const RECORD: &str = "@@";

/// The variable that turns [`child`] from a no-op into one case.
const CASE_VAR: &str = "OXIDELAKE_CORRUPT_PARQUET_CASE";
const DIR_VAR: &str = "OXIDELAKE_CORRUPT_PARQUET_DIR";

/// 4 GB of address space, the cap the 0.3.0 review ran its probes under.
const ADDRESS_SPACE_CAP: u64 = 4_000_000_000;

/// What a child may use at most, in KiB of peak RSS: the file is ~100 KB, so
/// anything near this means a declared length was believed.
const RSS_LIMIT_KIB: i64 = 512 * 1024;

const ROWS: i64 = 10_000;

/// The file `oxide gen-data` would write for one Int64 column: OxideLake's
/// writer settings, page statistics on, so the page index is present, and a
/// Bloom filter on the key.
fn write_valid(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..ROWS))],
    )
    .unwrap();
    let opts = ParquetWriteOptions {
        compression: Compression::None,
        ..Default::default()
    }
    .with_bloom_filter("k");
    write_parquet(path, schema, &[batch], &opts).unwrap();
}

/// The same column written as two row groups of ten pages each, so the
/// offset index has page locations worth corrupting.
fn write_paged(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_max_row_group_row_count(Some(5_000))
        .set_data_page_row_count_limit(500)
        .set_write_batch_size(500)
        .set_compression(parquet::basic::Compression::UNCOMPRESSED)
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_dictionary_enabled(false)
        .build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    for start in (0..ROWS).step_by(500) {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(start..start + 500))],
        )
        .unwrap();
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
}

fn write_case_base(case: &str, path: &Path) {
    if case.starts_with("page-") || case == "index-count" {
        write_paged(path);
    } else {
        write_valid(path);
    }
}

fn footer_len(file: &[u8]) -> usize {
    let n = file.len();
    u32::from_le_bytes(file[n - 8..n - 4].try_into().unwrap()) as usize
}

/// Replaces the footer with `md`, keeping every byte before it (data pages
/// and the page index the column chunks still point at). The kept bytes go
/// through the writer's own byte count, so a page index it writes from `md`
/// is pointed at where it really lands rather than at offset zero.
fn rewrite_footer(file: &[u8], md: &ParquetMetaData) -> Vec<u8> {
    let data_end = file.len() - 8 - footer_len(file);
    let mut out = Vec::with_capacity(file.len() + 4096);
    let mut tracked = TrackedWrite::new(&mut out);
    tracked.write_all(&file[..data_end]).unwrap();
    ParquetMetaDataWriter::new_with_tracked(tracked, md)
        .finish()
        .unwrap();
    out
}

fn metadata(file: &[u8], policy: PageIndexPolicy) -> ParquetMetaData {
    ParquetMetaDataReader::new()
        .with_page_index_policy(policy)
        .parse_and_finish(&Bytes::from(file.to_vec()))
        .unwrap()
}

/// Rewrites one field of row group 0, column 0.
fn patch_chunk(
    file: &[u8],
    patch: impl FnOnce(ColumnChunkMetaDataBuilder) -> ColumnChunkMetaDataBuilder,
) -> Vec<u8> {
    let mut builder =
        ParquetMetaDataBuilder::new_from_metadata(metadata(file, PageIndexPolicy::Skip));
    let mut row_groups = builder.take_row_groups();
    let mut rg = row_groups.remove(0).into_builder();
    let mut columns = rg.take_columns();
    let c0 = columns.remove(0);
    columns.insert(0, patch(c0.into_builder()).build().unwrap());
    row_groups.insert(0, rg.set_column_metadata(columns).build().unwrap());
    rewrite_footer(file, &builder.set_row_groups(row_groups).build())
}

/// Rewrites the page locations of row group 0, column 0; `patch` gets them
/// with the chunk's start.
fn patch_pages(file: &[u8], patch: impl FnOnce(&mut Vec<PageLocation>, i64)) -> Vec<u8> {
    let md = metadata(file, PageIndexPolicy::Required);
    let (start, _) = md.row_group(0).column(0).byte_range();
    let mut offset_index = md.offset_index().cloned().unwrap();
    patch(&mut offset_index[0][0].page_locations, start as i64);
    let md = ParquetMetaDataBuilder::new_from_metadata(md)
        .set_offset_index(Some(offset_index))
        .build();
    rewrite_footer(file, &md)
}

/// Thrift compact protocol, as much as hand-written footers need.
fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn zigzag(v: i64, out: &mut Vec<u8>) {
    varint(((v << 1) ^ (v >> 63)) as u64, out);
}

fn list_header(size: u64, element: u8, out: &mut Vec<u8>) {
    if size < 15 {
        out.push(((size as u8) << 4) | element);
    } else {
        out.push(0xF0 | element);
        varint(size, out);
    }
}

/// A schema element: a group of `children`, or the INT64 leaf `k`.
fn schema_element(name: &str, children: Option<i64>, out: &mut Vec<u8>) {
    let mut last = 0;
    let mut field = |id: i16, ty: u8, out: &mut Vec<u8>| {
        out.push((((id - last) as u8) << 4) | ty);
        last = id;
    };
    if children.is_none() {
        field(1, 5, out); // type: INT64
        zigzag(2, out);
    }
    field(3, 5, out); // repetition: REQUIRED
    zigzag(0, out);
    field(4, 8, out); // name
    varint(name.len() as u64, out);
    out.extend_from_slice(name.as_bytes());
    if let Some(children) = children {
        field(5, 5, out); // num_children
        zigzag(children, out);
    }
    out.push(0);
}

/// A whole file around a hand-written `FileMetaData`: magic, footer, its
/// length, magic. There are no data pages; none of these get that far.
fn file_with_footer(footer: &[u8]) -> Vec<u8> {
    let mut out = b"PAR1".to_vec();
    out.extend_from_slice(footer);
    out.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    out.extend_from_slice(b"PAR1");
    out
}

/// A footer whose schema is `elements`, followed by `row_groups` declared
/// row groups and nothing else.
fn footer(elements: &[u8], schema_len: u64, row_groups: Option<u64>) -> Vec<u8> {
    let mut out = vec![0x15]; // version
    zigzag(1, &mut out);
    out.push(0x19); // schema: list<struct>
    list_header(schema_len, 12, &mut out);
    out.extend_from_slice(elements);
    out.push(0x16); // num_rows
    zigzag(0, &mut out);
    if let Some(row_groups) = row_groups {
        out.push(0x19); // row_groups: list<struct>
        list_header(row_groups, 12, &mut out);
    }
    out
}

/// Builds the bytes of one case from a fresh valid file.
fn corrupt(case: &str, dir: &Path) -> Vec<u8> {
    let base = dir.join("base.parquet");
    write_case_base(case, &base);
    let file = std::fs::read(&base).unwrap();
    match case {
        "ok" | "page-ok" => file,
        // The footer is intact; the first page header is garbage, so the
        // file registers and only the scan can find out (#118).
        "bad-page" => {
            let mut out = file;
            out[4..36].fill(0xFF);
            out
        }
        // The three files of #109: the footer decodes, one length lies.
        "neg-len" => patch_chunk(&file, |c| c.set_total_compressed_size(-1)),
        // The same without a page index, so the page checks cannot catch it
        // on the chunk's behalf.
        "neg-len-bare" => patch_chunk(&file, |c| {
            c.set_total_compressed_size(-1)
                .set_offset_index_offset(None)
                .set_offset_index_length(None)
                .set_column_index_offset(None)
                .set_column_index_length(None)
        }),
        "big-len" => patch_chunk(&file, |c| c.set_total_compressed_size(2_000_000_000)),
        "huge-len" => patch_chunk(&file, |c| c.set_total_compressed_size(1 << 40)),
        // The start instead of the length.
        "neg-offset" => patch_chunk(&file, |c| {
            c.set_dictionary_page_offset(None).set_data_page_offset(-1)
        }),
        // A page index range declared past the end of the file.
        "index-len" => patch_chunk(&file, |c| c.set_column_index_length(Some(i32::MAX))),
        // A Bloom filter declared past the end of the file: an equality
        // predicate reads it whole.
        "bloom-len" => patch_chunk(&file, |c| c.set_bloom_filter_length(Some(i32::MAX))),
        // The footer's own length: more than the file holds.
        "footer-len" => {
            let mut out = file;
            let n = out.len();
            out[n - 8..n - 4].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
            out
        }
        // Page locations outside the chunk they belong to.
        "page-past-chunk" => patch_pages(&file, |pages, _| {
            pages[0].compressed_page_size = 1_000_000_000;
        }),
        "page-before-chunk" => patch_pages(&file, |pages, start| {
            pages[0].offset = start - 1_000;
        }),
        "page-negative" => patch_pages(&file, |pages, _| {
            pages[1].offset = -1;
        }),
        "page-rows-backwards" => patch_pages(&file, |pages, _| {
            pages[2].first_row_index = 1;
        }),
        // Counts the decoder reserves for before reading one element.
        "index-count" => {
            let md = metadata(&file, PageIndexPolicy::Skip);
            let at = md.row_group(0).column(0).offset_index_offset().unwrap() as usize;
            let mut out = file;
            // List header of `page_locations`: 2^31 - 1 pages, long form.
            out[at + 1..at + 7].copy_from_slice(&[0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0x07]);
            out
        }
        "row-group-count" => {
            let mut schema = Vec::new();
            schema_element("schema", Some(1), &mut schema);
            schema_element("k", None, &mut schema);
            file_with_footer(&footer(&schema, 2, Some(i32::MAX as u64)))
        }
        "schema-children" => {
            let mut schema = Vec::new();
            schema_element("schema", Some(i64::from(i32::MAX)), &mut schema);
            schema_element("k", None, &mut schema);
            file_with_footer(&footer(&schema, 2, Some(0)))
        }
        "schema-depth" => {
            const DEPTH: u64 = 100_000;
            let mut schema = Vec::new();
            for _ in 0..=DEPTH {
                schema_element("g", Some(1), &mut schema);
            }
            schema_element("k", None, &mut schema);
            file_with_footer(&footer(&schema, DEPTH + 2, Some(0)))
        }
        other => panic!("unknown case {other}"),
    }
}

fn outcome(result: &Result<(), EngineError>) -> String {
    match result {
        Ok(()) => "Ok".into(),
        Err(EngineError::Format(m)) => format!("Format {m}"),
        Err(EngineError::Io(e)) => format!("Io {e}"),
        Err(other) => format!("Other {other}"),
    }
}

async fn query(ctx: &SessionContext, sql: &str) -> Result<(), EngineError> {
    let batches = ctx
        .sql(sql)
        .await
        .map_err(classify_error)?
        .collect()
        .await
        .map_err(classify_error)?;
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    println!("{RECORD}ROWS {rows}");
    Ok(())
}

fn max_rss_kib() -> i64 {
    // SAFETY: getrusage writes into the zeroed struct it is given.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage.ru_maxrss
    }
}

/// The query shapes that reach a file differently: a full aggregate (scan
/// tasks spawned per partition), a `LIMIT` (read on the caller's thread), a
/// range the page index can narrow, and an equality the Bloom filter can.
const QUERIES: [(&str, &str); 4] = [
    ("SUM", "SELECT SUM(k) FROM t"),
    ("LIMIT", "SELECT k FROM t LIMIT 3"),
    ("RANGE", "SELECT k FROM t WHERE k BETWEEN 600 AND 700"),
    ("EQ", "SELECT k FROM t WHERE k = 4242"),
];

async fn query_all(ctx: &SessionContext) {
    for (key, sql) in QUERIES {
        println!("{RECORD}{key} {}", outcome(&query(ctx, sql).await));
    }
}

/// What a Ballista scheduler does with a query: plans it and stops. Its
/// executors run the plan with DataFusion's own reader, so the plan is the
/// last place the scheduler's guard can refuse a file.
async fn plan_all(ctx: &SessionContext) {
    for (key, sql) in QUERIES {
        let planned = async { ctx.sql(sql).await?.create_physical_plan().await }
            .await
            .map(drop)
            .map_err(classify_error);
        println!("{RECORD}{key} {}", outcome(&planned));
    }
}

/// The child half: a no-op in a normal test run, one case when the parent
/// re-executes this binary with [`CASE_VAR`] set.
///
/// `register:<case>` registers the case's file and, if registration lets it
/// through, queries it. `scan:<case>` registers a valid file and then
/// replaces it with the case, so only a check at scan time can catch it;
/// `dir:<case>` registers a directory holding a valid file and then adds the
/// case to it. `plan:<case>` is `scan:` in a scheduler's session, which only
/// plans.
#[test]
fn child() {
    let Ok(spec) = std::env::var(CASE_VAR) else {
        return;
    };
    let dir = PathBuf::from(std::env::var(DIR_VAR).unwrap());
    let (when, case) = spec.split_once(':').unwrap();
    let bytes = corrupt(case, &dir);
    let table = dir.join("table");
    std::fs::create_dir(&table).unwrap();
    let path = table.join("t.parquet");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let ctx = match when {
            "dir" => session_without_caches(),
            "plan" => scheduler_session(),
            _ => pruning_session(),
        };
        let registered = match when {
            "register" => {
                std::fs::write(&path, &bytes).unwrap();
                register_parquet_table(&ctx, "t", path.to_str().unwrap()).await
            }
            "scan" | "dir" | "plan" => {
                write_case_base(case, &path);
                let target = if when == "dir" { &table } else { &path };
                let registered = register_parquet_table(&ctx, "t", target.to_str().unwrap()).await;
                // Published the way writers publish: a rename into place.
                // The mtime is moved on explicitly. DataFusion keys its
                // metadata cache by size and mtime, and a same-size
                // replacement inside one timestamp tick would be served the
                // old, valid footer instead of being read at all.
                let staged = dir.join("staged");
                std::fs::write(&staged, &bytes).unwrap();
                File::options()
                    .write(true)
                    .open(&staged)
                    .unwrap()
                    .set_modified(std::time::SystemTime::now() + Duration::from_secs(60))
                    .unwrap();
                let published = if when == "dir" {
                    table.join("u.parquet")
                } else {
                    path.clone()
                };
                std::fs::rename(&staged, published).unwrap();
                registered
            }
            other => panic!("unknown phase {other}"),
        };
        println!("{RECORD}REGISTER {}", outcome(&registered));
        if registered.is_ok() {
            if when == "plan" {
                plan_all(&ctx).await;
            } else {
                query_all(&ctx).await;
            }
        }
    });
    println!("{RECORD}MAXRSS_KIB {}", max_rss_kib());
}

/// `pruning_session` without DataFusion's file-listing and metadata caches.
/// DataFusion 55 caches a directory's listing for the life of the session
/// (no TTL by default), so a file added after registration is never scanned
/// at all; with the cache off it is, and the scan-time check has to catch
/// it. Without the metadata cache a scan reads the footer alone, as it does
/// once an entry is evicted, and the page index is the guard's to load.
fn session_without_caches() -> SessionContext {
    let runtime = RuntimeEnvBuilder::new()
        .with_cache_manager(
            CacheManagerConfig::default()
                .with_list_files_cache_limit(0)
                .with_metadata_cache_limit(0),
        )
        .build_arc()
        .unwrap();
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(with_pruning(SessionConfig::new()))
        .with_runtime_env(runtime)
        .with_physical_optimizer_rule(Arc::new(ParquetScanGuard::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_local_store(&ctx, default_object_store());
    ctx
}

/// The session `oxidelake-runtime` builds for a Ballista scheduler, less
/// Ballista: the guard checks what planning read into the session's own
/// metadata cache.
fn scheduler_session() -> SessionContext {
    let runtime = RuntimeEnvBuilder::new().build_arc().unwrap();
    let guard = ParquetScanGuard::new()
        .with_metadata_cache(runtime.cache_manager.get_file_metadata_cache());
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(with_pruning(SessionConfig::new()))
        .with_runtime_env(runtime)
        .with_physical_optimizer_rule(Arc::new(guard))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_local_store(&ctx, default_object_store());
    ctx
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

impl Run {
    /// The value the child reported under `key`. The harness prints
    /// "test child ... " before the child's first line, so a record is
    /// found by its marker rather than at the start of a line.
    fn line(&self, key: &str) -> Option<&str> {
        let marker = format!("{RECORD}{key} ");
        self.stdout
            .lines()
            .find_map(|l| l.split_once(marker.as_str()).map(|(_, rest)| rest))
    }

    fn explain(&self) -> String {
        let stderr: Vec<&str> = self.stderr.lines().collect();
        let tail = stderr[stderr.len().saturating_sub(15)..].join("\n");
        format!(
            "status {:?}\n--- stdout\n{}\n--- stderr (tail)\n{tail}",
            self.status, self.stdout
        )
    }
}

/// Runs one case in a capped child and waits (bounded) for it.
fn run_capped(spec: &str) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["child", "--exact", "--nocapture", "--test-threads=1"])
        .env(CASE_VAR, spec)
        .env(DIR_VAR, dir.path())
        // One malloc arena per thread reserves 64 MiB of address space each;
        // under the cap that is space the case under test cannot use.
        .env("MALLOC_ARENA_MAX", "2")
        .env("RUST_BACKTRACE", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // SAFETY: setrlimit is async-signal-safe and affects only the child.
    unsafe {
        command.pre_exec(|| {
            let cap = libc::rlimit {
                rlim_cur: ADDRESS_SPACE_CAP,
                rlim_max: ADDRESS_SPACE_CAP,
            };
            if libc::setrlimit(libc::RLIMIT_AS, &cap) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let out = std::thread::spawn(move || {
        let mut s = String::new();
        stdout.read_to_string(&mut s).unwrap();
        s
    });
    let err = std::thread::spawn(move || {
        let mut s = String::new();
        stderr.read_to_string(&mut s).unwrap();
        s
    });
    // Generous: a loaded machine is slow, and a hung child is a failure, not
    // something to wait on forever.
    let deadline = Instant::now() + Duration::from_secs(300);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Run {
        status,
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
    }
}

/// The child exited on its own (no panic, no abort) inside the memory limit.
fn assert_clean(case: &str, run: &Run) {
    assert!(
        run.status.success(),
        "{case}: the child did not exit cleanly\n{}",
        run.explain()
    );
    let rss: i64 = run
        .line("MAXRSS_KIB")
        .unwrap_or_else(|| panic!("{case}: no MAXRSS line\n{}", run.explain()))
        .parse()
        .unwrap();
    assert!(
        rss < RSS_LIMIT_KIB,
        "{case}: peak RSS {rss} KiB for a ~100 KB file\n{}",
        run.explain()
    );
}

fn assert_format(case: &str, run: &Run, key: &str, needles: &[&str]) {
    let line = run
        .line(key)
        .unwrap_or_else(|| panic!("{case}: no {key} line\n{}", run.explain()));
    assert!(
        line.starts_with("Format "),
        "{case}: {key} was not a Format error: {line}\n{}",
        run.explain()
    );
    for needle in needles {
        assert!(
            line.contains(needle),
            "{case}: {key} does not name {needle:?}: {line}"
        );
    }
}

/// The harness itself: a valid file registers and queries in the child.
#[test]
fn a_valid_file_registers_and_scans_in_the_capped_child() {
    for spec in ["register:ok", "register:page-ok", "scan:ok", "dir:page-ok"] {
        let run = run_capped(spec);
        assert_clean(spec, &run);
        for key in ["REGISTER", "SUM", "LIMIT", "RANGE", "EQ"] {
            assert_eq!(run.line(key), Some("Ok"), "{spec} {key}\n{}", run.explain());
        }
    }
}

/// Corruption the footer cannot reveal is found by the scan, and comes back
/// classified (#118): before, two of the three query shapes returned
/// `EngineError::DataFusion(..)` because `classify_error` did not look
/// inside the wrapper a repartitioned scan puts around its error.
#[test]
fn a_corrupt_page_found_by_the_scan_is_a_format_error() {
    let run = run_capped("register:bad-page");
    assert_clean("bad-page", &run);
    assert_eq!(run.line("REGISTER"), Some("Ok"), "{}", run.explain());
    for key in ["SUM", "LIMIT", "RANGE"] {
        assert_format("bad-page", &run, key, &["struct field type"]);
    }
}

/// The three files of #109 and the same lie told by the other range fields:
/// refused at registration, naming the file, the row group and the column.
/// Before: -1 panicked the scan (exit 101), 2,000,000,000 cost 2 GB of RSS,
/// 2^40 aborted on a 1 TiB allocation, and the page-index length cost 2 GB
/// at registration itself.
#[test]
fn a_corrupt_column_chunk_is_refused_at_registration() {
    for case in [
        "neg-len",
        "neg-len-bare",
        "big-len",
        "huge-len",
        "neg-offset",
        "index-len",
        "bloom-len",
    ] {
        let spec = format!("register:{case}");
        let run = run_capped(&spec);
        assert_clean(&spec, &run);
        assert_format(
            &spec,
            &run,
            "REGISTER",
            &["t.parquet", "row group 0", "column k"],
        );
    }
}

/// Page locations are checked against the chunk they belong to. Before,
/// each of these panicked a scan task, and the first cost 1 GB of RSS.
#[test]
fn a_page_outside_its_chunk_is_refused_at_registration() {
    for case in [
        "page-past-chunk",
        "page-before-chunk",
        "page-negative",
        "page-rows-backwards",
    ] {
        let spec = format!("register:{case}");
        let run = run_capped(&spec);
        assert_clean(&spec, &run);
        assert_format(
            &spec,
            &run,
            "REGISTER",
            &["t.parquet", "row group 0", "column k", "page"],
        );
    }
}

/// Lies in the footer's own encoding, each of which the decoder believed
/// before reading a single element: a footer length past the file (a
/// subtraction overflow in the decoder), 2^31 - 1 row groups, a schema group
/// of 2^31 - 1 children and 2^31 - 1 page locations (allocations of hundreds
/// of GiB, so an abort), and 100,000 nested groups (a stack overflow).
#[test]
fn a_footer_that_declares_more_than_it_holds_is_refused_at_registration() {
    for (case, needle) in [
        ("footer-len", "claims"),
        ("row-group-count", "declares"),
        ("schema-children", "children"),
        ("schema-depth", "nests deeper"),
        ("index-count", "offset index"),
    ] {
        let spec = format!("register:{case}");
        let run = run_capped(&spec);
        assert_clean(&spec, &run);
        assert_format(&spec, &run, "REGISTER", &["t.parquet", needle]);
    }
}

/// A file replaced after its table was registered, or added to a registered
/// directory, is checked when the scan opens it (the session carries
/// `ParquetScanGuard`): a `Format` error naming the file (and the column,
/// where the scan got that far) instead of a panic, an abort or a 2 GB read.
/// `index-len` is refused one step earlier, by the bounded store, when
/// DataFusion reads the page index for statistics while planning.
#[test]
fn a_file_corrupted_after_registration_is_refused_by_the_scan() {
    for (spec, needle) in [
        ("scan:neg-len", "column k"),
        ("scan:neg-len-bare", "column k"),
        ("scan:big-len", "column k"),
        ("scan:huge-len", "column k"),
        ("scan:neg-offset", "column k"),
        ("scan:page-past-chunk", "column k"),
        ("scan:page-before-chunk", "column k"),
        ("scan:page-negative", "column k"),
        ("scan:index-len", "lie outside"),
        ("scan:bloom-len", "column k"),
        ("dir:neg-len", "column k"),
        ("dir:huge-len", "column k"),
        ("dir:page-before-chunk", "column k"),
    ] {
        let run = run_capped(spec);
        assert_clean(spec, &run);
        assert_eq!(
            run.line("REGISTER"),
            Some("Ok"),
            "{spec}\n{}",
            run.explain()
        );
        // A `LIMIT 3` over a directory stops in the valid file it reads
        // first, so it never opens the one added later.
        let keys: &[&str] = if spec.starts_with("dir:") {
            &["SUM", "RANGE", "EQ"]
        } else {
            &["SUM", "LIMIT", "RANGE", "EQ"]
        };
        for key in keys {
            assert_format(spec, &run, key, &[".parquet", needle]);
        }
    }
}

/// A cluster's executors scan with DataFusion's own reader, so its scheduler
/// checks, while it plans, the metadata planning has just read: a file
/// corrupted after registration fails the query there, before any executor
/// would read by its footer. Without that check these plans succeed.
#[test]
fn a_scheduler_refuses_a_file_corrupted_after_registration_while_it_plans() {
    for spec in [
        "plan:neg-len",
        "plan:huge-len",
        "plan:neg-offset",
        "plan:page-past-chunk",
    ] {
        let run = run_capped(spec);
        assert_clean(spec, &run);
        assert_eq!(
            run.line("REGISTER"),
            Some("Ok"),
            "{spec}\n{}",
            run.explain()
        );
        for key in ["SUM", "LIMIT", "RANGE", "EQ"] {
            assert_format(spec, &run, key, &["t.parquet", "row group 0", "column k"]);
        }
    }
}
