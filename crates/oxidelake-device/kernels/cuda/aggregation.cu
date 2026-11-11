// OxideLake CUDA kernels: SUM/COUNT/MIN/MAX grouped by one Int64 key (docs/SPEC.md §2.3).
//
// Groups live in an open-addressing table (power-of-two capacity, atomicCAS
// claim, state 0 empty / 1 claimed / 2 published). Aggregates are atomic
// reductions into per-slot accumulators; the NULL key is its own group held
// in dedicated accumulators (index 0 of the *_null arrays). The host reads the
// table back, compacts occupied slots and sorts by key (NULL group last), so
// the output matches the CPU reference exactly.
//
// Semantics: SUM/MIN/MAX skip NULL values and are NULL when a group has none
// (count == 0); COUNT counts non-NULL values. SUM over Int64 wraps on
// overflow (the unsigned atomicAdd), as stock DataFusion's SUM and the CPU
// reference do. MIN/MAX over Float64 follow IEEE 754 totalOrder, as Arrow's
// comparisons and the CPU reference do (see oxide_total_order_key).
//
// Compiled at runtime with NVRTC; there is no nvcc build step.

extern "C" __device__ __forceinline__ int oxide_ag_is_valid(const unsigned char* validity, unsigned int i) {
    return validity == 0 ? 1 : ((validity[i >> 3] >> (i & 7u)) & 1u);
}

extern "C" __device__ __forceinline__ unsigned int oxide_ag_hash(long long key, unsigned int capacity) {
    unsigned long long h = (unsigned long long)key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (unsigned int)(h & (unsigned long long)(capacity - 1u));
}

// Finds the slot holding `key` (inserting it when `insert` != 0). Returns
// 0xFFFFFFFF when the key is absent (lookup) or the table is full (insert).
//
// Publication protocol: the winner of the CAS stores the key, fences, then
// stores state 2. A thread that loses the CAS for a slot (or finds it in
// state 1) spins until the winner publishes, then compares the key. Both the
// spin and the key comparison read through `volatile` pointers: a plain load
// may be served from this SM's L1, which is not coherent, and a neighbouring
// probe can have cached the line while the slot was still empty. Comparing a
// stale key makes the loser probe on and insert the same key a second time,
// which surfaces as a duplicate group with count 0. The fence between the
// state load and the key load pairs with the writer's fence, so the key
// store is visible once state 2 is observed. Lookups (`insert` == 0) run in
// a later kernel launch, after every publication is complete, so they need
// neither the spin nor the fence.
//
// Lanes of one warp may wait on each other here, which requires independent
// thread scheduling (compute capability >= 7.0, Volta); the CUDA backend
// targets no older GPUs.
extern "C" __device__ __forceinline__ unsigned int oxide_ag_load_state(const unsigned int* p) {
    return *((volatile const unsigned int*)p);
}

extern "C" __device__ __forceinline__ long long oxide_ag_load_key(const long long* p) {
    return *((volatile const long long*)p);
}

extern "C" __device__ unsigned int oxide_ag_slot(long long key,
                                                 long long* g_keys,
                                                 unsigned int* g_state,
                                                 unsigned int capacity,
                                                 int insert) {
    unsigned int slot = oxide_ag_hash(key, capacity);
    for (unsigned int probes = 0; probes < capacity; ++probes) {
        unsigned int state = oxide_ag_load_state(&g_state[slot]);
        if (state == 0u) {
            if (!insert) return 0xFFFFFFFFu;
            if (atomicCAS(&g_state[slot], 0u, 1u) == 0u) {
                g_keys[slot] = key;
                __threadfence();
                g_state[slot] = 2u;
                return slot;
            }
            // Lost the race: another thread owns this slot; wait for its key.
            state = 1u;
        }
        if (state == 1u) {
            while (oxide_ag_load_state(&g_state[slot]) == 1u) { }
        }
        if (insert) __threadfence();
        if (oxide_ag_load_key(&g_keys[slot]) == key) return slot;
        slot = (slot + 1u) & (capacity - 1u);
    }
    return 0xFFFFFFFFu;
}

// Pass 1: insert every non-NULL key; flag the NULL group.
extern "C" __global__ void oxide_agg_insert_keys(const long long* __restrict__ keys,
                                                 const unsigned char* __restrict__ validity,
                                                 unsigned int n,
                                                 long long* __restrict__ g_keys,
                                                 unsigned int* __restrict__ g_state,
                                                 unsigned int capacity,
                                                 unsigned int* __restrict__ null_group_flag) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_ag_is_valid(validity, i)) { atomicOr(null_group_flag, 1u); return; }
    oxide_ag_slot(keys[i], g_keys, g_state, capacity, 1);
}

// Pass 2 kernels: one per (function, type). `counts` receives the non-NULL
// value count per group and doubles as COUNT.

extern "C" __global__ void oxide_agg_sum_i64(const long long* __restrict__ keys,
                                             const unsigned char* __restrict__ kvalid,
                                             const long long* __restrict__ values,
                                             const unsigned char* __restrict__ vvalid,
                                             unsigned int n,
                                             long long* __restrict__ g_keys,
                                             unsigned int* __restrict__ g_state,
                                             unsigned int capacity,
                                             long long* __restrict__ sums,
                                             unsigned int* __restrict__ counts,
                                             long long* __restrict__ null_sum,
                                             unsigned int* __restrict__ null_count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_ag_is_valid(vvalid, i)) return;
    long long v = values[i];
    if (!oxide_ag_is_valid(kvalid, i)) {
        atomicAdd((unsigned long long*)null_sum, (unsigned long long)v);
        atomicAdd(null_count, 1u);
        return;
    }
    unsigned int slot = oxide_ag_slot(keys[i], g_keys, g_state, capacity, 0);
    if (slot == 0xFFFFFFFFu) return;
    atomicAdd((unsigned long long*)&sums[slot], (unsigned long long)v);
    atomicAdd(&counts[slot], 1u);
}

extern "C" __global__ void oxide_agg_sum_f64(const long long* __restrict__ keys,
                                             const unsigned char* __restrict__ kvalid,
                                             const double* __restrict__ values,
                                             const unsigned char* __restrict__ vvalid,
                                             unsigned int n,
                                             long long* __restrict__ g_keys,
                                             unsigned int* __restrict__ g_state,
                                             unsigned int capacity,
                                             double* __restrict__ sums,
                                             unsigned int* __restrict__ counts,
                                             double* __restrict__ null_sum,
                                             unsigned int* __restrict__ null_count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_ag_is_valid(vvalid, i)) return;
    double v = values[i];
    if (!oxide_ag_is_valid(kvalid, i)) {
        atomicAdd(null_sum, v);
        atomicAdd(null_count, 1u);
        return;
    }
    unsigned int slot = oxide_ag_slot(keys[i], g_keys, g_state, capacity, 0);
    if (slot == 0xFFFFFFFFu) return;
    atomicAdd(&sums[slot], v);
    atomicAdd(&counts[slot], 1u);
}

extern "C" __global__ void oxide_agg_minmax_i64(const long long* __restrict__ keys,
                                                const unsigned char* __restrict__ kvalid,
                                                const long long* __restrict__ values,
                                                const unsigned char* __restrict__ vvalid,
                                                unsigned int n,
                                                long long* __restrict__ g_keys,
                                                unsigned int* __restrict__ g_state,
                                                unsigned int capacity,
                                                long long* __restrict__ mins,
                                                long long* __restrict__ maxs,
                                                unsigned int* __restrict__ counts,
                                                long long* __restrict__ null_min,
                                                long long* __restrict__ null_max,
                                                unsigned int* __restrict__ null_count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_ag_is_valid(vvalid, i)) return;
    long long v = values[i];
    if (!oxide_ag_is_valid(kvalid, i)) {
        atomicMin(null_min, v);
        atomicMax(null_max, v);
        atomicAdd(null_count, 1u);
        return;
    }
    unsigned int slot = oxide_ag_slot(keys[i], g_keys, g_state, capacity, 0);
    if (slot == 0xFFFFFFFFu) return;
    atomicMin(&mins[slot], v);
    atomicMax(&maxs[slot], v);
    atomicAdd(&counts[slot], 1u);
}

// IEEE 754 totalOrder as a signed 64-bit integer: flipping every bit but
// the sign of a negative double makes bit patterns compare, as signed
// integers, the way Rust's `f64::total_cmp` and Arrow's float comparisons
// order the values: -NaN < -inf < ... < -0.0 < 0.0 < ... < inf < NaN. MIN and
// MAX then reduce with the native 64-bit atomicMin/atomicMax, and the host
// decodes the result with the same transform (it is its own inverse). The
// IEEE comparisons this replaces ignored NaN — an all-NaN group reported the
// +inf/-inf the accumulators started at — and could not tell -0.0 from 0.0.
extern "C" __device__ __forceinline__ long long oxide_total_order_key(double v) {
    unsigned long long bits = (unsigned long long)__double_as_longlong(v);
    unsigned long long flip = (0ull - (bits >> 63)) >> 1;
    return (long long)(bits ^ flip);
}

extern "C" __global__ void oxide_agg_minmax_f64(const long long* __restrict__ keys,
                                                const unsigned char* __restrict__ kvalid,
                                                const double* __restrict__ values,
                                                const unsigned char* __restrict__ vvalid,
                                                unsigned int n,
                                                long long* __restrict__ g_keys,
                                                unsigned int* __restrict__ g_state,
                                                unsigned int capacity,
                                                long long* __restrict__ mins,
                                                long long* __restrict__ maxs,
                                                unsigned int* __restrict__ counts,
                                                long long* __restrict__ null_min,
                                                long long* __restrict__ null_max,
                                                unsigned int* __restrict__ null_count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_ag_is_valid(vvalid, i)) return;
    long long v = oxide_total_order_key(values[i]);
    if (!oxide_ag_is_valid(kvalid, i)) {
        atomicMin(null_min, v);
        atomicMax(null_max, v);
        atomicAdd(null_count, 1u);
        return;
    }
    unsigned int slot = oxide_ag_slot(keys[i], g_keys, g_state, capacity, 0);
    if (slot == 0xFFFFFFFFu) return;
    atomicMin(&mins[slot], v);
    atomicMax(&maxs[slot], v);
    atomicAdd(&counts[slot], 1u);
}

// COUNT(value): non-NULL values per group (vvalid may be null = all valid).
extern "C" __global__ void oxide_agg_count(const long long* __restrict__ keys,
                                           const unsigned char* __restrict__ kvalid,
                                           const unsigned char* __restrict__ vvalid,
                                           unsigned int n,
                                           long long* __restrict__ g_keys,
                                           unsigned int* __restrict__ g_state,
                                           unsigned int capacity,
                                           unsigned int* __restrict__ counts,
                                           unsigned int* __restrict__ null_count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!oxide_ag_is_valid(vvalid, i)) return;
    if (!oxide_ag_is_valid(kvalid, i)) { atomicAdd(null_count, 1u); return; }
    unsigned int slot = oxide_ag_slot(keys[i], g_keys, g_state, capacity, 0);
    if (slot == 0xFFFFFFFFu) return;
    atomicAdd(&counts[slot], 1u);
}
