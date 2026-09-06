// OxideLake Metal kernels: L2 and cosine distance between a query vector and a
// FixedSizeList<Float32> column (docs/SPEC.md §2.3).
//
// One SIMD-group (32 lanes) per row; lanes stride over the dimensions and the
// partial sums are reduced with simd_sum. Output validity equals input
// validity, so the host reuses the input bitmap and only values are written.
// metric: 0 = L2, 1 = cosine distance (NaN when either norm is zero).
// Dispatch with threadgroups of 256 threads = 8 rows per threadgroup.
//
// Compiled at runtime with MTLDevice::newLibraryWithSource; no Xcode step.

#include <metal_stdlib>
using namespace metal;

inline bool oxide_vd_is_valid(device const uchar* validity, uint has_validity, uint i) {
    return has_validity == 0 ? true : (((validity[i >> 3] >> (i & 7)) & 1) != 0);
}

kernel void oxide_vec_distance(device const float* vectors     [[buffer(0)]],
                               device const uchar* validity    [[buffer(1)]],
                               constant uint& has_validity     [[buffer(2)]],
                               constant uint& n                [[buffer(3)]],
                               constant uint& dim              [[buffer(4)]],
                               device const float* query       [[buffer(5)]],
                               constant int& metric            [[buffer(6)]],
                               device float* out               [[buffer(7)]],
                               uint tid  [[thread_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    uint row = tid / 32;
    if (row >= n) return;
    if (!oxide_vd_is_valid(validity, has_validity, row)) {
        if (lane == 0) out[row] = 0.0f;   // masked by the validity bitmap
        return;
    }
    device const float* v = vectors + (ulong)row * dim;
    float dot = 0.0f, nv = 0.0f, nq = 0.0f, dist = 0.0f;
    for (uint k = lane; k < dim; k += 32) {
        float a = v[k];
        float b = query[k];
        if (metric == 0) {
            float d = a - b;
            dist += d * d;
        } else {
            dot += a * b;
            nv += a * a;
            nq += b * b;
        }
    }
    if (metric == 0) {
        dist = simd_sum(dist);
        if (lane == 0) out[row] = sqrt(dist);
    } else {
        dot = simd_sum(dot);
        nv = simd_sum(nv);
        nq = simd_sum(nq);
        if (lane == 0) {
            float na = sqrt(nv), nb = sqrt(nq);
            out[row] = (na == 0.0f || nb == 0.0f) ? NAN : 1.0f - dot / (na * nb);
        }
    }
}
