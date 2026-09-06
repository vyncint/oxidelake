// OxideLake CUDA kernels: L2 and cosine distance between a query vector and a
// FixedSizeList<Float32> column (docs/SPEC.md §2.3).
//
// One warp per row: lanes stride over the dimensions, partial sums are reduced
// with warp shuffles, lane 0 writes the result. The query vector is staged in
// dynamic shared memory (dim * 4 bytes, host-provided). Output validity equals
// input validity (a NULL vector yields a NULL distance), so the host reuses the
// input bitmap and the kernel writes only the values.
//
// metric: 0 = L2 (Euclidean), 1 = cosine distance = 1 - cos(theta); cosine is
// NaN when either norm is zero, matching the CPU reference.
//
// Compiled at runtime with NVRTC; there is no nvcc build step.

extern "C" __device__ __forceinline__ int oxide_vd_is_valid(const unsigned char* validity, unsigned int i) {
    return validity == 0 ? 1 : ((validity[i >> 3] >> (i & 7u)) & 1u);
}

extern "C" __device__ __forceinline__ float oxide_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Launch with blockDim.x a multiple of 32; grid.x = ceil(n / (blockDim.x / 32)).
// Dynamic shared memory: dim * sizeof(float).
extern "C" __global__ void oxide_vec_distance(const float* __restrict__ vectors,
                                              const unsigned char* __restrict__ validity,
                                              unsigned int n,
                                              unsigned int dim,
                                              const float* __restrict__ query,
                                              int metric,
                                              float* __restrict__ out) {
    extern __shared__ float q[];
    for (unsigned int k = threadIdx.x; k < dim; k += blockDim.x) q[k] = query[k];
    __syncthreads();

    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp_in_block = threadIdx.x >> 5;
    unsigned int warps_per_block = blockDim.x >> 5;
    unsigned int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= n) return;

    if (!oxide_vd_is_valid(validity, row)) {
        if (lane == 0) out[row] = 0.0f;   // value is masked by the validity bitmap
        return;
    }

    const float* v = vectors + (unsigned long long)row * dim;
    float dot = 0.0f, nv = 0.0f, nq = 0.0f, dist = 0.0f;
    for (unsigned int k = lane; k < dim; k += 32u) {
        float a = v[k];
        float b = q[k];
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
        dist = oxide_warp_sum(dist);
        if (lane == 0) out[row] = sqrtf(dist);
    } else {
        dot = oxide_warp_sum(dot);
        nv = oxide_warp_sum(nv);
        nq = oxide_warp_sum(nq);
        if (lane == 0) {
            float na = sqrtf(nv), nb = sqrtf(nq);
            out[row] = (na == 0.0f || nb == 0.0f) ? __int_as_float(0x7fc00000) : 1.0f - dot / (na * nb);
        }
    }
}
