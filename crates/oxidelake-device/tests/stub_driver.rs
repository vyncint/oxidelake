//! An NVIDIA driver without NVRTC (#106): a driver-only host, or a container
//! that has `libcuda` injected by the NVIDIA container toolkit and nothing
//! else from the CUDA toolkit. Detection used to probe `libcuda` alone, so it
//! selected CUDA there, and the first kernel compile then panicked inside
//! cudarc ("Unable to dynamically load the "nvrtc" shared library") — no
//! query could fall back to the CPU.
//!
//! No GPU is needed. The test builds a stub `libcuda.so.1` (the driver entry
//! points detection reaches, each succeeding and reporting one device) and
//! re-runs this binary with the stub first on `LD_LIBRARY_PATH`, which the
//! loader reads once at process start — hence the child process.
//!
//! Linux with a C compiler (`$CC`, default `cc`); run with
//! `cargo test -p oxidelake-device --features cuda --test stub_driver`.

#![cfg(all(feature = "cuda", target_os = "linux"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::Command;

use oxidelake_core::{BackendKind, EngineError};
use oxidelake_device::HardwareDetector;
use oxidelake_device::cuda::CudaBackend;

/// Set in the child: the scenario below only means something under the stub.
const CHILD: &str = "OXIDE_TEST_STUB_DRIVER";

/// What the reviewer's repro used: every call cudarc makes for
/// `CudaContext::device_count` / `CudaContext::new` and their teardown.
const STUB_SOURCE: &str = r#"
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
static char fake_ctx;
static CUcontext current = 0;
CUresult cuInit(unsigned int flags) { (void)flags; return 0; }
CUresult cuDeviceGetCount(int *count) { *count = 1; return 0; }
CUresult cuDeviceGet(CUdevice *d, int ordinal) { *d = ordinal; return 0; }
CUresult cuDevicePrimaryCtxRetain(CUcontext *p, CUdevice d) { (void)d; *p = &fake_ctx; return 0; }
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice d) { (void)d; return 0; }
CUresult cuDeviceGetAttribute(int *v, int a, CUdevice d) { (void)a; (void)d; *v = 0; return 0; }
CUresult cuCtxGetCurrent(CUcontext *p) { *p = current; return 0; }
CUresult cuCtxSetCurrent(CUcontext c) { current = c; return 0; }
CUresult cuGetErrorString(CUresult e, const char **s) { (void)e; *s = "stub"; return 0; }
CUresult cuGetErrorName(CUresult e, const char **s) { (void)e; *s = "STUB"; return 0; }
"#;

#[test]
fn a_driver_without_nvrtc_selects_the_cpu_and_never_panics() {
    let dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("stub-libcuda-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("stub.c");
    std::fs::write(&source, STUB_SOURCE).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let built = match Command::new(&cc)
        .args(["-shared", "-fPIC", "-o"])
        .arg(dir.join("libcuda.so.1"))
        .arg(&source)
        .output()
    {
        Ok(out) => out,
        Err(e) => {
            eprintln!("skipped: no C compiler `{cc}` to build the stub libcuda ({e})");
            return;
        }
    };
    assert!(
        built.status.success(),
        "building the stub libcuda failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    // cudarc tries `libcuda.so` before `libcuda.so.1`.
    std::os::unix::fs::symlink("libcuda.so.1", dir.join("libcuda.so")).unwrap();

    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "stub_driver_scenario",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("LD_LIBRARY_PATH", &dir)
        .env(CHILD, "1")
        .env_remove("OXIDE_BACKEND")
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "the stub-driver scenario failed:\n{stdout}\n{stderr}"
    );
    // A filter that matched nothing would also exit 0.
    assert!(
        stdout.contains("stub driver scenario: ok"),
        "the scenario did not run:\n{stdout}\n{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
#[ignore = "the child half of a_driver_without_nvrtc_selects_the_cpu_and_never_panics"]
fn stub_driver_scenario() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    assert!(
        CudaBackend::driver_present(),
        "the stub libcuda was not found on LD_LIBRARY_PATH"
    );
    // A machine that has a real NVRTC cannot reproduce the missing-NVRTC shape.
    if CudaBackend::nvrtc_present() {
        println!("stub driver scenario: ok (skipped: this machine has a real libnvrtc)");
        return;
    }
    let expected = "libnvrtc";

    let why = CudaBackend::probe(0).unwrap_err();
    assert!(why.to_string().contains(expected), "{why}");
    assert!(!CudaBackend::is_available());
    assert!(!HardwareDetector::availability().cuda);

    // Auto-detection falls back to the CPU; it used to select CUDA here.
    let selected = HardwareDetector::select_with(None).unwrap();
    assert_eq!(selected.kind(), BackendKind::CpuSimd);
    assert_eq!(HardwareDetector::detected_kind(), BackendKind::CpuSimd);

    // An explicit request is a typed error naming what is missing.
    let err = HardwareDetector::select_with(Some("cuda")).unwrap_err();
    assert!(
        matches!(
            err,
            EngineError::Device {
                backend: BackendKind::Cuda,
                ..
            }
        ),
        "{err}"
    );
    let message = err.to_string();
    assert!(message.contains("explicitly requested"), "{message}");
    assert!(message.contains(expected), "{message}");

    let err = CudaBackend::new(0).unwrap_err();
    assert!(err.to_string().contains(expected), "{err}");
    println!("stub driver scenario: ok");
}
