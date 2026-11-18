// OxideLake CUDA kernels: inner hash join on one Int64 key (docs/SPEC.md §2.3).
//
// The build side is indexed once per operator (#45), on the host, from the
// CPU reference's JoinTable (src/join_slots.rs), and uploaded: an
// open-addressing table over a power-of-two capacity with one slot per
// DISTINCT key (#108), each holding (start, count) into a row-index array
// that groups the build rows by key, in build order. count == 0 marks an
// empty slot (every key in the table has at least one row), so a probe stops
// at its own key or at the first empty slot and a duplicated build key costs
// it nothing. Nothing here writes the table, so no kernel needs an atomic.
//
// Pipeline (host orchestrated, src/cuda/ops.rs):
//   per probe batch
//     1. oxide_hj_count  : per probe row, its key's (start, count); a NULL or
//                          absent key gets count 0
//     2. (host) exclusive scan of the counts -> offsets, total
//   per output chunk of at most one batch (#95)
//     3. oxide_hj_expand : (left_idx, right_idx) of matches [first, first + rows),
//                          in probe-row order, then build order within a row
//     4. oxide_gather_* (filter_project module) per output column
//
// Compiled at runtime with NVRTC; there is no nvcc build step.

extern "C" __device__ __forceinline__ int oxide_hj_is_valid(const unsigned char* validity, unsigned int i) {
    return validity == 0 ? 1 : ((validity[i >> 3] >> (i & 7u)) & 1u);
}

// The host places each key at the slot this returns (`slot_hash` in
// src/join_slots.rs); the two must agree bit for bit, or a probe starts its
// walk somewhere its key was never put.
extern "C" __device__ __forceinline__ unsigned int oxide_hj_hash(long long key, unsigned int capacity) {
    unsigned long long h = (unsigned long long)key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (unsigned int)(h & (unsigned long long)(capacity - 1u));
}

extern "C" __global__ void oxide_hj_count(const long long* __restrict__ probe_keys,
                                          const unsigned char* __restrict__ validity,
                                          unsigned int n,
                                          const long long* __restrict__ t_keys,
                                          const unsigned int* __restrict__ t_starts,
                                          const unsigned int* __restrict__ t_counts,
                                          unsigned int capacity,
                                          unsigned int* __restrict__ counts,
                                          unsigned int* __restrict__ starts) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int count = 0u;
    unsigned int start = 0u;
    if (oxide_hj_is_valid(validity, i)) {
        long long key = probe_keys[i];
        unsigned int slot = oxide_hj_hash(key, capacity);
        for (unsigned int probes = 0u; probes < capacity; ++probes) {
            unsigned int c = t_counts[slot];
            if (c == 0u) break;
            if (t_keys[slot] == key) {
                count = c;
                start = t_starts[slot];
                break;
            }
            slot = (slot + 1u) & (capacity - 1u);
        }
    }
    counts[i] = count;
    starts[i] = start;
}

// One thread per output row. `offsets` is the exclusive scan of the counts
// (n entries, offsets[0] == 0); the host launches only for
// first + rows <= total, so every p below is a real match.
extern "C" __global__ void oxide_hj_expand(const unsigned long long* __restrict__ offsets,
                                           const unsigned int* __restrict__ starts,
                                           unsigned int n,
                                           const unsigned int* __restrict__ t_rows,
                                           unsigned int n_rows,
                                           unsigned long long first,
                                           unsigned int rows,
                                           unsigned int* __restrict__ left_idx,
                                           unsigned int* __restrict__ right_idx) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows) return;
    unsigned long long p = first + t;
    // Match p belongs to the LAST probe row whose offset is <= p: offsets[0]
    // is 0, and a row with no matches never wins, because the row after it
    // starts at the same offset.
    unsigned int lo = 0u;
    unsigned int hi = n;
    while (hi - lo > 1u) {
        unsigned int mid = lo + (hi - lo) / 2u;
        if (offsets[mid] <= p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    unsigned long long at = (unsigned long long)starts[lo] + (p - offsets[lo]);
    left_idx[t] = lo;
    // The host checked start + count <= n_rows for every key before
    // uploading the table; the guard only keeps a broken invariant in bounds.
    right_idx[t] = at < (unsigned long long)n_rows ? t_rows[at] : 0u;
}
