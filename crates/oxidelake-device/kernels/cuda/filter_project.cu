// OxideLake CUDA kernels: fused filter + projection (docs/SPEC.md §2.3).
//
// Data layout (see oxidelake-device `columns.rs`): fixed-width values tightly packed
// from row 0; an optional validity bitmap, LSB-first, one bit per row (a null
// pointer means "all rows valid"). Every kernel bounds-checks its global
// accesses; grid sizes are computed on the host.
//
// Pipeline (host orchestrated, all launches on one stream):
//   1. oxide_compare_{i64,f64}  : one byte mask per comparison leaf
//   2. oxide_mask_and           : conjunction of leaf masks
//   3. oxide_block_count        : selected rows per 256-thread block
//   4. (host) exclusive scan of block counts -> block offsets, total
//   5. oxide_scatter_indices    : stable compaction into row indices
//   6. oxide_gather_fixed / oxide_gather_validity per projected column
//
// Compiled at runtime with NVRTC; there is no nvcc build step.

#define OXIDE_BLOCK 256u

extern "C" __device__ __forceinline__ int oxide_is_valid(const unsigned char* validity, unsigned int i) {
    return validity == 0 ? 1 : ((validity[i >> 3] >> (i & 7u)) & 1u);
}

// op: 0 == , 1 < , 2 <= , 3 > , 4 >=
extern "C" __global__ void oxide_compare_i64(const long long* __restrict__ values,
                                             const unsigned char* __restrict__ validity,
                                             int op,
                                             long long literal,
                                             unsigned char* __restrict__ mask,
                                             unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_is_valid(validity, i)) { mask[i] = 0; return; }
    long long v = values[i];
    int r;
    switch (op) {
        case 0: r = v == literal; break;
        case 1: r = v <  literal; break;
        case 2: r = v <= literal; break;
        case 3: r = v >  literal; break;
        default: r = v >= literal; break;
    }
    mask[i] = (unsigned char)r;
}

extern "C" __global__ void oxide_compare_f64(const double* __restrict__ values,
                                             const unsigned char* __restrict__ validity,
                                             int op,
                                             double literal,
                                             unsigned char* __restrict__ mask,
                                             unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_is_valid(validity, i)) { mask[i] = 0; return; }
    double v = values[i];
    int r;
    switch (op) {
        case 0: r = v == literal; break;
        case 1: r = v <  literal; break;
        case 2: r = v <= literal; break;
        case 3: r = v >  literal; break;
        default: r = v >= literal; break;
    }
    mask[i] = (unsigned char)r;
}

extern "C" __global__ void oxide_mask_and(const unsigned char* __restrict__ a,
                                          const unsigned char* __restrict__ b,
                                          unsigned char* __restrict__ out,
                                          unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] & b[i];
}

// One count per block of OXIDE_BLOCK rows; launch with blockDim.x == OXIDE_BLOCK.
extern "C" __global__ void oxide_block_count(const unsigned char* __restrict__ mask,
                                             unsigned int n,
                                             unsigned int* __restrict__ block_counts) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int selected = (i < n) ? (mask[i] != 0) : 0;
    int count = __syncthreads_count(selected);
    if (threadIdx.x == 0) block_counts[blockIdx.x] = (unsigned int)count;
}

// Stable stream compaction: block-local exclusive scan (shared memory) plus the
// host-computed block offset. Launch with blockDim.x == OXIDE_BLOCK.
extern "C" __global__ void oxide_scatter_indices(const unsigned char* __restrict__ mask,
                                                 unsigned int n,
                                                 const unsigned int* __restrict__ block_offsets,
                                                 unsigned int* __restrict__ out_indices) {
    __shared__ unsigned int scan[OXIDE_BLOCK];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int flag = (i < n) ? (unsigned int)(mask[i] != 0) : 0u;
    scan[threadIdx.x] = flag;
    __syncthreads();
    // Hillis-Steele inclusive scan.
    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        unsigned int add = (threadIdx.x >= offset) ? scan[threadIdx.x - offset] : 0u;
        __syncthreads();
        scan[threadIdx.x] += add;
        __syncthreads();
    }
    if (flag) {
        unsigned int local = scan[threadIdx.x] - 1u;   // exclusive position
        out_indices[block_offsets[blockIdx.x] + local] = i;
    }
}

// Gathers fixed-width rows: dst[j] = src[indices[j]], `width` bytes per row.
extern "C" __global__ void oxide_gather_fixed(const unsigned char* __restrict__ src,
                                              unsigned int width,
                                              const unsigned int* __restrict__ indices,
                                              unsigned int n_out,
                                              unsigned char* __restrict__ dst) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const unsigned char* s = src + (unsigned long long)indices[j] * width;
    unsigned char* d = dst + (unsigned long long)j * width;
    if ((width & 7u) == 0u) {
        const unsigned long long* s8 = (const unsigned long long*)s;
        unsigned long long* d8 = (unsigned long long*)d;
        for (unsigned int k = 0; k < width / 8u; ++k) d8[k] = s8[k];
    } else if ((width & 3u) == 0u) {
        const unsigned int* s4 = (const unsigned int*)s;
        unsigned int* d4 = (unsigned int*)d;
        for (unsigned int k = 0; k < width / 4u; ++k) d4[k] = s4[k];
    } else {
        for (unsigned int k = 0; k < width; ++k) d[k] = s[k];
    }
}

// Gathers validity bits: one thread per output byte (8 output rows).
extern "C" __global__ void oxide_gather_validity(const unsigned char* __restrict__ src_validity,
                                                 const unsigned int* __restrict__ indices,
                                                 unsigned int n_out,
                                                 unsigned char* __restrict__ dst_validity) {
    unsigned int byte = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n_bytes = (n_out + 7u) / 8u;
    if (byte >= n_bytes) return;
    unsigned char out = 0;
    for (unsigned int bit = 0; bit < 8u; ++bit) {
        unsigned int j = byte * 8u + bit;
        if (j >= n_out) break;
        if (oxide_is_valid(src_validity, indices[j])) out |= (unsigned char)(1u << bit);
    }
    dst_validity[byte] = out;
}
