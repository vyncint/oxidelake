// OxideLake CUDA kernels: inner hash join on one Int64 key (docs/SPEC.md §2.3).
//
// Open-addressing table with linear probing over a power-of-two capacity.
// Slot state is a 32-bit word claimed with atomicCAS (0 = empty, 1 = occupied);
// duplicate build keys occupy separate slots, so a probe walks the chain until
// it reaches an empty slot and reports every match. NULL keys never match and
// are never inserted.
//
// Pipeline (host orchestrated):
//   1. oxide_hj_build : insert build-side rows
//   2. oxide_hj_count : matches per probe row
//   3. (host) exclusive scan of counts -> offsets, total
//   4. oxide_hj_write : (left_idx, right_idx) pairs in probe-row order
//   5. oxide_gather_* (filter_project module) per output column
//
// Compiled at runtime with NVRTC; there is no nvcc build step.

extern "C" __device__ __forceinline__ int oxide_hj_is_valid(const unsigned char* validity, unsigned int i) {
    return validity == 0 ? 1 : ((validity[i >> 3] >> (i & 7u)) & 1u);
}

extern "C" __device__ __forceinline__ unsigned int oxide_hj_hash(long long key, unsigned int capacity) {
    unsigned long long h = (unsigned long long)key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (unsigned int)(h & (unsigned long long)(capacity - 1u));
}

extern "C" __global__ void oxide_hj_build(const long long* __restrict__ keys,
                                          const unsigned char* __restrict__ validity,
                                          unsigned int n,
                                          long long* __restrict__ t_keys,
                                          unsigned int* __restrict__ t_rows,
                                          unsigned int* __restrict__ t_state,
                                          unsigned int capacity) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_hj_is_valid(validity, i)) return;
    long long key = keys[i];
    unsigned int slot = oxide_hj_hash(key, capacity);
    for (unsigned int probes = 0; probes < capacity; ++probes) {
        if (atomicCAS(&t_state[slot], 0u, 1u) == 0u) {
            t_keys[slot] = key;
            t_rows[slot] = i;
            __threadfence();
            t_state[slot] = 2u;   // published
            return;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    // Table full: cannot happen when capacity >= 2 * n (host guarantees it).
}

extern "C" __global__ void oxide_hj_count(const long long* __restrict__ probe_keys,
                                          const unsigned char* __restrict__ validity,
                                          unsigned int n,
                                          const long long* __restrict__ t_keys,
                                          const unsigned int* __restrict__ t_state,
                                          unsigned int capacity,
                                          unsigned int* __restrict__ counts) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int count = 0;
    if (oxide_hj_is_valid(validity, i)) {
        long long key = probe_keys[i];
        unsigned int slot = oxide_hj_hash(key, capacity);
        for (unsigned int probes = 0; probes < capacity; ++probes) {
            unsigned int state = t_state[slot];
            if (state == 0u) break;
            if (state == 2u && t_keys[slot] == key) ++count;
            slot = (slot + 1u) & (capacity - 1u);
        }
    }
    counts[i] = count;
}

extern "C" __global__ void oxide_hj_write(const long long* __restrict__ probe_keys,
                                          const unsigned char* __restrict__ validity,
                                          unsigned int n,
                                          const long long* __restrict__ t_keys,
                                          const unsigned int* __restrict__ t_rows,
                                          const unsigned int* __restrict__ t_state,
                                          unsigned int capacity,
                                          const unsigned int* __restrict__ offsets,
                                          unsigned int* __restrict__ left_idx,
                                          unsigned int* __restrict__ right_idx) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_hj_is_valid(validity, i)) return;
    long long key = probe_keys[i];
    unsigned int out = offsets[i];
    unsigned int slot = oxide_hj_hash(key, capacity);
    for (unsigned int probes = 0; probes < capacity; ++probes) {
        unsigned int state = t_state[slot];
        if (state == 0u) break;
        if (state == 2u && t_keys[slot] == key) {
            left_idx[out] = i;
            right_idx[out] = t_rows[slot];
            ++out;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
}
