// OxideLake Metal kernels: fused filter + projection (docs/SPEC.md §2.3).
//
// Same data layout and pipeline as the CUDA version: byte masks per comparison
// leaf, conjunction, per-threadgroup counts, host exclusive scan, stable
// scatter of row indices, then per-column gathers. Metal buffers cannot be
// null, so `has_validity` selects between the bitmap and "all valid".
// Threadgroup size for count/scatter is 256.
//
// Compiled at runtime with MTLDevice::newLibraryWithSource; no Xcode step.

#include <metal_stdlib>
using namespace metal;

constant uint OXIDE_TG = 256;

inline bool oxide_is_valid(device const uchar* validity, uint has_validity, uint i) {
    return has_validity == 0 ? true : (((validity[i >> 3] >> (i & 7)) & 1) != 0);
}

// op: 0 == , 1 < , 2 <= , 3 > , 4 >=
kernel void oxide_compare_i64(device const long* values      [[buffer(0)]],
                              device const uchar* validity   [[buffer(1)]],
                              constant uint& has_validity    [[buffer(2)]],
                              constant int& op               [[buffer(3)]],
                              constant long& literal         [[buffer(4)]],
                              device uchar* mask             [[buffer(5)]],
                              constant uint& n               [[buffer(6)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    if (!oxide_is_valid(validity, has_validity, i)) { mask[i] = 0; return; }
    long v = values[i];
    bool r;
    switch (op) {
        case 0: r = v == literal; break;
        case 1: r = v <  literal; break;
        case 2: r = v <= literal; break;
        case 3: r = v >  literal; break;
        default: r = v >= literal; break;
    }
    mask[i] = r ? 1 : 0;
}

// Metal has no double on many devices: Float64 columns are compared on the
// host-side CPU path (the backend reports Unsupported for them).

kernel void oxide_mask_and(device const uchar* a   [[buffer(0)]],
                           device const uchar* b   [[buffer(1)]],
                           device uchar* out       [[buffer(2)]],
                           constant uint& n        [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] = a[i] & b[i];
}

kernel void oxide_block_count(device const uchar* mask         [[buffer(0)]],
                              constant uint& n                 [[buffer(1)]],
                              device uint* block_counts        [[buffer(2)]],
                              uint i   [[thread_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint gid [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint total;
    if (tid == 0) atomic_store_explicit(&total, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint flag = (i < n) ? (mask[i] != 0 ? 1u : 0u) : 0u;
    uint warp_sum = simd_sum(flag);
    if (simd_is_first()) atomic_fetch_add_explicit(&total, warp_sum, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) block_counts[gid] = atomic_load_explicit(&total, memory_order_relaxed);
}

// Stable compaction: in-threadgroup exclusive scan (SIMD prefix + threadgroup
// memory) plus the host-computed threadgroup offset.
kernel void oxide_scatter_indices(device const uchar* mask           [[buffer(0)]],
                                  constant uint& n                   [[buffer(1)]],
                                  device const uint* block_offsets   [[buffer(2)]],
                                  device uint* out_indices           [[buffer(3)]],
                                  uint i    [[thread_position_in_grid]],
                                  uint tid  [[thread_index_in_threadgroup]],
                                  uint gid  [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint simd_totals[OXIDE_TG / 32];
    uint flag = (i < n) ? (mask[i] != 0 ? 1u : 0u) : 0u;
    uint local = simd_prefix_exclusive_sum(flag);
    uint simd_total = simd_sum(flag);
    if (lane == 0) simd_totals[sg] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0;
    for (uint s = 0; s < sg; ++s) base += simd_totals[s];
    if (flag) out_indices[block_offsets[gid] + base + local] = i;
}

kernel void oxide_gather_fixed(device const uchar* src        [[buffer(0)]],
                               constant uint& width           [[buffer(1)]],
                               device const uint* indices     [[buffer(2)]],
                               constant uint& n_out           [[buffer(3)]],
                               device uchar* dst              [[buffer(4)]],
                               uint j [[thread_position_in_grid]]) {
    if (j >= n_out) return;
    device const uchar* s = src + (ulong)indices[j] * width;
    device uchar* d = dst + (ulong)j * width;
    for (uint k = 0; k < width; ++k) d[k] = s[k];
}

kernel void oxide_gather_validity(device const uchar* src_validity [[buffer(0)]],
                                  constant uint& has_validity      [[buffer(1)]],
                                  device const uint* indices       [[buffer(2)]],
                                  constant uint& n_out             [[buffer(3)]],
                                  device uchar* dst_validity       [[buffer(4)]],
                                  uint byte [[thread_position_in_grid]]) {
    uint n_bytes = (n_out + 7) / 8;
    if (byte >= n_bytes) return;
    uchar out = 0;
    for (uint bit = 0; bit < 8; ++bit) {
        uint j = byte * 8 + bit;
        if (j >= n_out) break;
        if (oxide_is_valid(src_validity, has_validity, indices[j])) out |= (uchar)(1u << bit);
    }
    dst_validity[byte] = out;
}
