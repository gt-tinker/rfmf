// ---------------------------------------------------------------------------
// The GPU sweep, C independent chains per launch: block b anneals chain b
// with its own (t0, seed), state slice (stride nPad), AnnealState and beta
// table; chains share only the read-only CSR.  No inter-block sync anywhere,
// so excess blocks simply run in waves.  delta(v) == spin[v]*S[v] is computed
// on the fly from the live neighbour-sum cache S (no deltas array needed).
//
// Per-sweep pipeline: SELECT (score every vertex, per-group argmax into P
// slots) -> barrier -> FLIP (read winners' PRE-SWEEP spin/S, compute corr[],
// update currentVal/bestVal, flip, scatter S) -> barrier.
//
// DO NOT "SIMPLIFY" FLIP's barriers: the serial kernel reads live S[pick]
// between scatters, so group order is load-bearing.  Scattering all winners
// in parallel then reading S[pick] double-counts winner-winner edges --
// currentVal silently drifts.  corr[] buys that ordering back analytically.
// ---------------------------------------------------------------------------

#include "anneal_gpu.cuh"
#include "philox.cuh"
#include "tuning.hpp"

#include <algorithm>
#include <type_traits>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

namespace gpu3 {

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t e_ = (err);                                                \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(e_));                              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// group(v) = v % P, the hottest scalar op in SELECT.  Both operands fit in
// 32 bits and unsigned 32-bit modulo is several times cheaper than signed
// 64-bit; identical result since v >= 0 and P > 0.
__device__ __forceinline__ uint32_t groupOf(int64_t v, int64_t P) {
    return static_cast<uint32_t>(v) % static_cast<uint32_t>(P);
}

// atomicAdd on the S cache.  int64 goes through the unsigned long long
// overload -- two's-complement addition is bit-identical, so this is exact.
__device__ __forceinline__ void atomicAddS(int32_t* p, int64_t v) {
    atomicAdd(p, static_cast<int32_t>(v));
}
__device__ __forceinline__ void atomicAddS(int64_t* p, int64_t v) {
    atomicAdd(reinterpret_cast<unsigned long long*>(p),
              static_cast<unsigned long long>(v));
}

// Warp-shuffle block sum (the reduce stage's multi-warp path): fold each
// warp, park one partial per warp in smem, ONE barrier, warp 0 folds the
// rest.  Exact (integer adds commute).  Works for ANY blockDim; every thread
// must reach this.  RETURN VALUE IS MEANINGFUL FOR tid 0 ONLY.
__device__ __forceinline__ int64_t blockReduceSumShuffle(int64_t v, int64_t* smem) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nWarps = (static_cast<int>(blockDim.x) + 31) >> 5;
    const int wCount = min(32, static_cast<int>(blockDim.x) - (warp << 5));
    const unsigned mask = (wCount == 32) ? 0xFFFFFFFFu : ((1u << wCount) - 1u);
    for (int off = 16; off > 0; off >>= 1) {
        const int64_t other = __shfl_down_sync(mask, v, off, 32);
        if (lane + off < wCount) v += other;
    }
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    int64_t total = 0;
    if (warp == 0) {
        total = (lane < nWarps) ? smem[lane] : 0;
        const int w0Count = min(32, static_cast<int>(blockDim.x));
        const unsigned m0 = (w0Count == 32) ? 0xFFFFFFFFu : ((1u << w0Count) - 1u);
        for (int off = 16; off > 0; off >>= 1) {
            const int64_t other = __shfl_down_sync(m0, total, off, 32);
            if (lane + off < w0Count) total += other;
        }
    }
    return total;
}

// Warp-0 register fold (the reduce stage's one-warp path).  Exact ONLY when
// every nonzero partial lives in warp 0 (the caller must guarantee it).  No
// smem, no barrier.  RETURN VALUE IS MEANINGFUL FOR tid 0 ONLY.
__device__ __forceinline__ int64_t warpFoldW0(int64_t v) {
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int w0 = min(32, static_cast<int>(blockDim.x));
        const unsigned m0 = (w0 == 32) ? 0xFFFFFFFFu : ((1u << w0) - 1u);
        for (int off = 16; off > 0; off >>= 1) {
            const int64_t other = __shfl_down_sync(m0, v, off, 32);
            if (tid + off < w0) v += other;
        }
    }
    return v;
}

// Load 8 consecutive S values with vector loads (SELECT's blocked8 map).  The
// base pointer is 8-element aligned by construction (v0 multiple of 8,
// arrays padded to a multiple of 8 elements), satisfying int4/longlong2.
__device__ __forceinline__ void loadS8(const int32_t* p, int32_t out[8]) {
    const int4 a = *reinterpret_cast<const int4*>(p);
    const int4 b = *reinterpret_cast<const int4*>(p + 4);
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
}
__device__ __forceinline__ void loadS8(const int64_t* p, int64_t out[8]) {
    const longlong2 a = *reinterpret_cast<const longlong2*>(p);
    const longlong2 b = *reinterpret_cast<const longlong2*>(p + 2);
    const longlong2 c = *reinterpret_cast<const longlong2*>(p + 4);
    const longlong2 d = *reinterpret_cast<const longlong2*>(p + 6);
    out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
    out[4] = c.x; out[5] = c.y; out[6] = d.x; out[7] = d.y;
}


// ---------------------------------------------------------------------------
// FLIP -- one block, three internally-barriered stages (see the file header).
// ---------------------------------------------------------------------------

// corr[gb] = sum over EARLIER winners ga < gb adjacent to b of
// 2*w(a,b)*newSpin_a.  The triangular bound IS serial's group order, so this
// analytic sum replaces the serialization -- a pure schedule change.
// winnerGroup[v] = group whose winner is v, or -1.
__device__ __forceinline__ void computeCorr(const DevCsr& g, int64_t P,
                                            const int64_t* mPick, const int64_t* mSpin,
                                            const int32_t* winnerGroup, int64_t* corr,
                                            int tid, int nt) {
    // One thread per winner, O(deg(b)): every earlier adjacent winner appears
    // in b's neighbour list.
    for (int64_t gb = tid; gb < P; gb += nt) {
        int64_t c = 0;
        const int64_t b = mPick[gb];
        if (b >= 0) {
            for (uint32_t i = g.offsets[b]; i < g.offsets[b + 1]; ++i) {
                const int64_t ga = winnerGroup[g.edgeTo[i]];
                if (ga >= 0 && ga < gb)
                    c += 2 * static_cast<int64_t>(g.edgeWeight[i]) * (-mSpin[ga]);
            }
        }
        corr[gb] = c;
    }
}

// corr via one WARP per winner (CorrMode::Warpwalk): the lanes
// partition the same list the walk scans, so the sum is bit-identical.
// gb = 0 is skipped outright (corr[0] == 0 by the triangular bound).
// Handles ANY blockDim; loops containing shuffles are warp-uniform, so no
// lane can miss a shuffle.
__device__ __forceinline__ void computeCorrWarp(const DevCsr& g, int64_t P,
                                                const int64_t* mPick, const int64_t* mSpin,
                                                const int32_t* winnerGroup, int64_t* corr,
                                                int tid, int nt) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nWarps = (nt + 31) >> 5;
    const int wCount = min(32, nt - (warp << 5));
    const unsigned mask = (wCount == 32) ? 0xFFFFFFFFu : ((1u << wCount) - 1u);
    for (int64_t gb = warp; gb < P; gb += nWarps) {
        if (gb == 0) {
            if (lane == 0) corr[0] = 0;
            continue;
        }
        const int64_t b = mPick[gb];
        int64_t c = 0;
        if (b >= 0) {
            const uint32_t lo = g.offsets[b], hi = g.offsets[b + 1];
            for (uint32_t i = lo + lane; i < hi; i += wCount) {
                const int64_t ga = winnerGroup[g.edgeTo[i]];
                if (ga >= 0 && ga < gb)
                    c += 2 * static_cast<int64_t>(g.edgeWeight[i]) * (-mSpin[ga]);
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const int64_t other = __shfl_down_sync(mask, c, off, 32);
            if (lane + off < wCount) c += other;
        }
        if (lane == 0) corr[gb] = c;
    }
}


// ---------------------------------------------------------------------------
// mcKernel -- the single-block persistent chain kernel, one block PER CHAIN.
// The body is the single-chain kernel verbatim except the chain prologue,
// which is what lets the single-chain correctness argument carry over.
//
// Two fusions, both pure schedule changes: group slots are reset by the
// PREVIOUS sweep's FLIP as it reads them, and the bestSpin snapshot rides
// along in SELECT's spin[] read when the previous sweep improved.
//
// THE EPILOGUE IS NOT OPTIONAL: the fused snapshot is one sweep late, so if
// the FINAL sweep improves there is no next SELECT to carry the copy.
// ---------------------------------------------------------------------------

template <class SType, bool kPacked, CorrMode kCorr>
__global__ void mcKernel(DevCsr g, int8_t* spinAll, int8_t* bestSpinAll,
                         SType* SAll, const int64_t* betaRunEndAll,
                         const int32_t* betaRunValAll,
                         const int64_t* chainRunOff, const uint64_t* phaseP,
                         const uint64_t* phaseIters, int64_t nPhases,
                         const uint64_t* solverSeeds, AnnealState* stAll,
                         int64_t maxP, int32_t* winnerGroupAll, int64_t nPad) {
    extern __shared__ unsigned char smemRaw[];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int64_t n = g.n;

    // ---- Chain prologue: the ONLY multi-chain logic in the kernel. ----
    // Block b is chain b: offset every mutable pointer by b's slice (stride
    // nPad, multiple of 8 so vector loads stay aligned), pick b's beta table
    // and Philox key.  From here down the body is the single-chain kernel.
    const int64_t chain = blockIdx.x;
    int8_t* spin = spinAll + chain * nPad;
    int8_t* bestSpin = bestSpinAll + chain * nPad;
    SType* S = SAll + chain * nPad;
    int32_t* winnerGroup = winnerGroupAll + chain * nPad;
    AnnealState* const st = stAll + chain;
    const int64_t* betaEnds = betaRunEndAll + chainRunOff[chain];
    const int32_t* betaVals = betaRunValAll + chainRunOff[chain];
    const uint64_t solverSeed = solverSeeds[chain];

    // Shared layout: slots | mPick | mSpin | mDelta | corr | reduce | idSlots
    //                | rowLo | cum
    // (idSlots: int64 score path only -- costs a few maxP*4 bytes
    // regardless, not worth a second layout.)
    unsigned long long* slots = reinterpret_cast<unsigned long long*>(smemRaw);
    int64_t* mPick  = reinterpret_cast<int64_t*>(slots + maxP);
    int64_t* mSpin  = mPick + maxP;
    int64_t* mDelta = mSpin + maxP;
    int64_t* corr   = mDelta + maxP;
    int64_t* red    = corr + maxP;
    int32_t* idSlots = reinterpret_cast<int32_t*>(red + nt);
    long long* scoreSlots = reinterpret_cast<long long*>(slots);  // aliases slots

    const Philox2 key = philoxKeyFromSeed(solverSeed);

    // Initial S recompute, and slots primed for the first sweep.
    for (int64_t v = tid; v < n; v += nt) {
        int64_t acc = 0;
        for (uint32_t i = g.offsets[v]; i < g.offsets[v + 1]; ++i)
            acc += static_cast<int64_t>(g.edgeWeight[i]) * spin[g.edgeTo[i]];
        S[v] = static_cast<SType>(acc);
    }
    for (int64_t gi = tid; gi < maxP; gi += nt) {
        if (kPacked) slots[gi] = kPackedEmpty;
        else { scoreSlots[gi] = LLONG_MIN; idSlots[gi] = INT32_MAX; }
    }
    __syncthreads();

    uint64_t step = 0;
    // Run-length beta cursor (host_common.h), advanced by every thread in
    // lockstep.  Runs are >= 1 step long, so a single `if` per sweep can
    // never fall behind; tables hold >= 1 run, so entry-0 reads are in bounds.
    int64_t betaRun = 0;
    uint64_t betaNext = static_cast<uint64_t>(betaEnds[0]);
    int32_t beta_fixed = betaVals[0];
    for (int64_t p = 0; p < nPhases; ++p) {
        const int64_t P = static_cast<int64_t>(phaseP[p]);
        if (P == 0) continue;
        for (uint64_t it = 0; it < phaseIters[p]; ++it) {
            if (step >= betaNext) {
                ++betaRun;
                betaNext = static_cast<uint64_t>(betaEnds[betaRun]);
                beta_fixed = betaVals[betaRun];
            }
            ++step;
            const int improved = st->improved;

            // ---- SELECT (+ fused bestSpin snapshot) ----
            //
            // Blocked8 map: 8 consecutive vertices per thread -- one Philox
            // call per block of 8, vectorized spin/S loads.  Private
            // pre-reduction: when (8*nt) % P == 0, lane j only ever feeds
            // ONE group, so it keeps a private running max and does a single
            // atomic at the end.  Bit-exact either way (max is associative
            // and commutative); the fallback is exact for the rest.  acc[j]
            // is indexed only by the fully unrolled j so it stays in
            // registers (an indexed accumulator would go to local memory).
            {
                const bool privReduce = ((8LL * static_cast<int64_t>(nt)) % P) == 0;
                unsigned long long accP[8];
                long long accW[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) { accP[j] = kPackedEmpty; accW[j] = LLONG_MIN; }
                const int64_t nBlk = (n + 7) >> 3;
                for (int64_t b = tid; b < nBlk; b += nt) {
                    const int64_t v0 = b << 3;
                    const unsigned long long sp8 =
                        *reinterpret_cast<const unsigned long long*>(spin + v0);
                    if (improved)   // spin is stable during SELECT
                        *reinterpret_cast<unsigned long long*>(bestSpin + v0) = sp8;
                    SType s8[8];
                    loadS8(S + v0, s8);
                    const Philox4 blk = philox4x32_10(gumbelBlockCounter(step, b), key);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int64_t v = v0 + j;
                        if (v < n) {
                            const int8_t sv = static_cast<int8_t>(sp8 >> (8 * j));
                            const int64_t delta =
                                static_cast<int64_t>(sv) * static_cast<int64_t>(s8[j]);
                            const int32_t gum = gumbelFromBitsG(
                                gumbelBitsFromBlock(blk, static_cast<uint32_t>(j)));
                            if (kPacked) {
                                const int32_t score = gum + beta_fixed * static_cast<int32_t>(delta);
                                const unsigned long long k2 =
                                    packScoreId(score, static_cast<int32_t>(v));
                                if (privReduce) { if (k2 > accP[j]) accP[j] = k2; }
                                else atomicMax(&slots[groupOf(v, P)], k2);
                            } else {
                                const int64_t s = static_cast<int64_t>(gum) +
                                                  static_cast<int64_t>(beta_fixed) * delta;
                                if (privReduce) { if (s > accW[j]) accW[j] = static_cast<long long>(s); }
                                else atomicMax(&scoreSlots[groupOf(v, P)], static_cast<long long>(s));
                            }
                        }
                    }
                }
                if (privReduce) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        // Every vertex lane j saw has group (8*tid + j) % P;
                        // untouched lanes still hold the empty sentinel.
                        const int64_t v = 8LL * tid + j;
                        if (kPacked) {
                            if (accP[j] != kPackedEmpty) atomicMax(&slots[groupOf(v, P)], accP[j]);
                        } else {
                            if (accW[j] != LLONG_MIN) atomicMax(&scoreSlots[groupOf(v, P)], accW[j]);
                        }
                    }
                }
            }
            __syncthreads();

            // Int64 score path: score + id won't fit one 64-bit atomic, so the
            // id is resolved in a second pass (Philox regenerates the score).
            if (!kPacked) {
                {
                    const int64_t nBlk = (n + 7) >> 3;
                    for (int64_t b = tid; b < nBlk; b += nt) {
                        const int64_t v0 = b << 3;
                        const unsigned long long sp8 =
                            *reinterpret_cast<const unsigned long long*>(spin + v0);
                        SType s8[8];
                        loadS8(S + v0, s8);
                        const Philox4 blk = philox4x32_10(gumbelBlockCounter(step, b), key);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int64_t v = v0 + j;
                            if (v < n) {
                                const int64_t delta =
                                    static_cast<int64_t>(static_cast<int8_t>(sp8 >> (8 * j))) *
                                    static_cast<int64_t>(s8[j]);
                                const int32_t gum = gumbelFromBitsG(
                                    gumbelBitsFromBlock(blk, static_cast<uint32_t>(j)));
                                const int64_t s = static_cast<int64_t>(gum) +
                                                  static_cast<int64_t>(beta_fixed) * delta;
                                if (s == static_cast<int64_t>(scoreSlots[groupOf(v, P)]))
                                    atomicMin(&idSlots[groupOf(v, P)], static_cast<int32_t>(v));
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // ---- FLIP stage 1: read winners' PRE-SWEEP spin/S, clear slots ----
            // Clearing here (same thread that read slot gi) lets the next
            // SELECT start with no reset pass and no extra barrier.
            for (int64_t gi = tid; gi < P; gi += nt) {
                int64_t pick;
                if (kPacked) {
                    const unsigned long long s = slots[gi];
                    slots[gi] = kPackedEmpty;
                    pick = (s == kPackedEmpty) ? -1 : static_cast<int64_t>(unpackId(s));
                } else {
                    pick = (idSlots[gi] == INT32_MAX) ? -1 : static_cast<int64_t>(idSlots[gi]);
                    scoreSlots[gi] = LLONG_MIN;
                    idSlots[gi] = INT32_MAX;
                }
                mPick[gi] = pick;
                if (pick >= 0) {
                    const int64_t sp = static_cast<int64_t>(spin[pick]);
                    mSpin[gi] = sp;
                    mDelta[gi] = sp * static_cast<int64_t>(S[pick]);
                } else {
                    mSpin[gi] = 0;
                    mDelta[gi] = 0;
                }
            }
            // Publish the winner->group map corr[] walks; taken down again
            // below so the array stays all -1 between sweeps.
            for (int64_t gi = tid; gi < P; gi += nt)
                if (mPick[gi] >= 0) winnerGroup[mPick[gi]] = static_cast<int32_t>(gi);
            __syncthreads();

            // ---- FLIP stage 2: corr[] ----
            if constexpr (kCorr == CorrMode::Warpwalk)
                computeCorrWarp(g, P, mPick, mSpin, winnerGroup, corr, tid, nt);
            else
                computeCorr(g, P, mPick, mSpin, winnerGroup, corr, tid, nt);
            __syncthreads();

            for (int64_t gi = tid; gi < P; gi += nt)
                if (mPick[gi] >= 0) winnerGroup[mPick[gi]] = -1;

            // ---- FLIP stage 3: currentVal / bestVal ----
            int64_t local = 0;
            for (int64_t gi = tid; gi < P; gi += nt)
                if (mPick[gi] >= 0) local += mDelta[gi] + mSpin[gi] * corr[gi];
            int64_t total;
            // Partials live only in threads tid < min(P, nt); when that
            // fits one warp the sum is a register fold.  P and nt are
            // block-uniform, so the fallback branch is barrier-safe.
            if (min(static_cast<int64_t>(nt), P) <= 32)
                total = warpFoldW0(local);                   // valid for tid 0 only
            else
                total = blockReduceSumShuffle(local, red);   // valid for tid 0 only
            if (tid == 0) {
                st->currentVal += total;
                if (st->currentVal > st->bestVal) {
                    st->bestVal = st->currentVal;
                    st->improved = 1;
                } else {
                    st->improved = 0;
                }
            }

            // ---- FLIP stage 4: apply flips, scatter S ----
            for (int64_t gi = tid; gi < P; gi += nt) {
                const int64_t pick = mPick[gi];
                if (pick >= 0) spin[pick] = static_cast<int8_t>(-mSpin[gi]);
            }
            {
                // Flat scatter: scatter the CONCATENATION of the winners'
                // rows -- one flattened winner-edge list, balanced across
                // threads regardless of degree skew.
                uint32_t* rowLo = reinterpret_cast<uint32_t*>(idSlots + maxP);
                uint32_t* cum   = rowLo + maxP;   // maxP + 1 entries
                for (int64_t gi = tid; gi < P; gi += nt) {
                    const int64_t pick = mPick[gi];
                    uint32_t lo = 0, len = 0;
                    if (pick >= 0) {
                        lo = g.offsets[pick];
                        len = g.offsets[pick + 1] - lo;
                    }
                    rowLo[gi] = lo;
                    cum[gi + 1] = len;
                }
                __syncthreads();
                if (tid == 0) {
                    // O(P) scan on one thread: noise at P <= 64.
                    cum[0] = 0;
                    for (int64_t gi = 0; gi < P; ++gi) cum[gi + 1] += cum[gi];
                }
                __syncthreads();
                const uint32_t total = cum[P];
                for (uint32_t e = tid; e < total; e += nt) {
                    // Largest gi with cum[gi] <= e; empty rows never land.
                    uint32_t glo = 0, ghi = static_cast<uint32_t>(P);
                    while (ghi - glo > 1) {
                        const uint32_t mid = (glo + ghi) >> 1;
                        if (cum[mid] <= e) glo = mid; else ghi = mid;
                    }
                    const uint32_t i = rowLo[glo] + (e - cum[glo]);
                    const int64_t x = g.edgeTo[i];
                    const int64_t w2 = 2 * static_cast<int64_t>(g.edgeWeight[i]) * (-mSpin[glo]);
                    atomicAddS(&S[x], w2);
                }
            }
            __syncthreads();
        }
    }

    // The epilogue the fused snapshot requires (see the header comment).
    if (st->improved)
        for (int64_t v = tid; v < n; v += nt) bestSpin[v] = spin[v];
}


// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

namespace {

struct DevBufs {
    uint64_t* phaseP = nullptr;
    uint64_t* phaseIters = nullptr;
    uint32_t* offsets = nullptr;      // CSR: shared read-only by every chain
    int32_t*  edgeTo = nullptr;
    int32_t*  edgeWeight = nullptr;
    int8_t*   spin = nullptr;         // C x nPad, chain c at offset c*nPad
    int8_t*   bestSpin = nullptr;     // C x nPad
    void*     S = nullptr;            // C x nPad
    int64_t*  betaRunEnd = nullptr;   // concatenated run-length beta tables:
    int32_t*  betaRunVal = nullptr;   //   one (end, value) list per DISTINCT t0
    int64_t*  chainRunOff = nullptr;  // chain -> its table's first run (C entries)
    uint64_t* solverSeeds = nullptr;  // chain -> Philox key seed (C entries)
    int32_t*  winnerGroup = nullptr;  // C x nPad: v -> group whose winner is v, else -1
    AnnealState* st = nullptr;        // C entries
};

void freeAll(DevBufs& d) {
    for (void* p : {(void*)d.phaseP, (void*)d.phaseIters,
                    (void*)d.offsets, (void*)d.edgeTo, (void*)d.edgeWeight,
                    (void*)d.spin, (void*)d.bestSpin, d.S,
                    (void*)d.betaRunEnd, (void*)d.betaRunVal, (void*)d.chainRunOff,
                    (void*)d.solverSeeds, (void*)d.winnerGroup, (void*)d.st})
        if (p) cudaFree(p);
}

// One launch, C chains: chain c is persistent block c of mcKernel.  No
// inter-block sync, so C may exceed co-resident capacity (waves).
// The vertex count travels inside dg, so it is not a separate parameter.
template <class SType>
void runMC(DevBufs& d, DevCsr dg, int64_t nPad, int64_t C,
           int64_t nPhases, bool packed, CorrMode corr,
           int64_t maxP, int tpb, float* kernelMs) {
    // slots | mPick | mSpin | mDelta | corr  = 5 * maxP * 8
    // reduce scratch                          = tpb * 8
    // idSlots                                 = maxP * 4
    // rowLo + cum (Flat scatter prefix table) = (2 * maxP + 1) * 4
    auto smemFor = [&](int t) {
        return static_cast<size_t>(maxP) * 5 * sizeof(int64_t) +
               static_cast<size_t>(t) * sizeof(int64_t) +
               static_cast<size_t>(maxP) * sizeof(int32_t) +
               (2 * static_cast<size_t>(maxP) + 1) * sizeof(uint32_t);
    };

    // Query the LAUNCH device, not ordinal 0 -- a heterogeneous multi-GPU
    // host could otherwise mis-check the smem budget.
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int maxSmem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

    const size_t smem = smemFor(tpb);
    if (smem > static_cast<size_t>(maxSmem)) {
        std::fprintf(stderr,
                     "Error: needs %zu B of shared memory per chain block (maxP=%lld,\n"
                     "tpb=%d) but this device offers %d B. Use a smaller max P.\n",
                     smem, static_cast<long long>(maxP), tpb, maxSmem);
        std::exit(EXIT_FAILURE);
    }

    auto launch = [&](auto kern) {
        // Clamp tpb to what THIS instantiation can launch (register counts
        // vary per template combo; an over-ask would fail outright).
        // Geometry is a pure schedule knob, so clamping changes nothing else.
        cudaFuncAttributes attr{};
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
        int tpbEff = tpb;
        if (tpbEff > attr.maxThreadsPerBlock) {
            tpbEff = attr.maxThreadsPerBlock;
            std::fprintf(stderr,
                         "note: tpb %d exceeds this kernel variant's launchable max of %d "
                         "threads (register-limited); clamped to %d.\n",
                         tpb, attr.maxThreadsPerBlock, tpbEff);
        }
        const size_t smemL = smemFor(tpbEff);
        // Opt in to >48 KB shared memory when the layout needs it.
        CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smemL)));
        cudaEvent_t evStart, evStop;
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));
        CUDA_CHECK(cudaEventRecord(evStart));
        // Plain launch, grid = C: no cooperative launch needed.
        kern<<<static_cast<unsigned int>(C), tpbEff, smemL>>>(
                               dg, d.spin, d.bestSpin, static_cast<SType*>(d.S),
                               d.betaRunEnd, d.betaRunVal, d.chainRunOff,
                               d.phaseP, d.phaseIters, nPhases,
                               d.solverSeeds, d.st, maxP, d.winnerGroup, nPad);
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(kernelMs, evStart, evStop));
        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));
    };

    // All template instantiations are bit-identical; they differ only in
    // schedule.  Compile-time dispatch so no combination's codegen can
    // perturb another's.
    using CM = CorrMode;
    auto pick = [&](auto packedTag, auto corrTag) {
        launch(mcKernel<SType, decltype(packedTag)::value, decltype(corrTag)::value>);
    };
    auto pickCorr = [&](auto packedTag) {
        if (corr == CM::Walk) pick(packedTag, std::integral_constant<CM, CM::Walk>{});
        else                  pick(packedTag, std::integral_constant<CM, CM::Warpwalk>{});
    };
    if (packed) pickCorr(std::integral_constant<bool, true>{});
    else        pickCorr(std::integral_constant<bool, false>{});
}

template <class SType>
void runDispatch(const anneallib::AnnealCsr& ig,
                 int64_t B,
                 const std::vector<std::vector<int8_t>>& initialSpins,
                 const std::vector<double>& t0s,
                 const std::vector<size_t>& phases,
                 const std::vector<uint64_t>& iters,
                 const std::vector<uint64_t>& solverSeeds,
                 const BetaRunTables& beta,
                 const LaunchConfig& cfg,
                 std::vector<std::vector<int8_t>>& endSpins,
                 std::vector<std::vector<int8_t>>& bestSpins,
                 float* kernelMs,
                 std::vector<int64_t>* kernelBestVals,
                 std::vector<int64_t>* kernelFinalVals) {
    const int64_t n = ig.numVertices();
    const int64_t m2 = static_cast<int64_t>(ig.numEdges()) * 2;
    const int64_t C = static_cast<int64_t>(t0s.size());

    int64_t maxP = 0;
    for (size_t ph : phases) maxP = std::max<int64_t>(maxP, static_cast<int64_t>(ph));

    // The int32 gate, evaluated at min(t0s) (worst chain): one verdict picks
    // the instantiation for the whole grid (see host_common.h).  B is the
    // caller's, not a second maxWeightedDegree(ig) scan -- so this verdict and
    // the score-path the caller reported are the same verdict by construction.
    const int64_t totalSteps = annealTotalSteps(iters);
    const double minT0 = *std::min_element(t0s.begin(), t0s.end());
    const bool packed = annealScoreSafeInt32(totalSteps, minT0, B);

    // Threads per block, two regimes split at the live SM count.
    // C <= #SMs: per-chain-latency rule.  C > #SMs: tpb is a co-residency
    // knob (smaller blocks pack more chains per SM).  Output-invariant.
    int tpb = 0;
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        int sms = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        // Boundaries are device-queried; which tpb wins on each side is the
        // measured V100 record (tuning.hpp).
        if (C > sms) {
            tpb = (n > tuning::blocked8CoResBigMinN)
                      ? tuning::tpbCoResidentBlocked8BigN
                      : tuning::tpbCoResidentBlocked8;
        } else if (cfg.corr == CorrMode::Warpwalk) {
            tpb = tuning::tpbWarpwalk;
        } else {
            tpb = (n <= tuning::tpbWalkSmallMaxN) ? tuning::tpbWalkSmall
                                                  : tuning::tpbWalkLarge;
        }
    }
    // Auditable tpb resolution, like the "state:" line.
    std::fprintf(stderr, "tpb: %d\n", tpb);

    // Flatten the CSR for upload; shared read-only by every chain -- the one
    // allocation that does not scale with C.
    std::vector<uint32_t> hOffsets(n + 1, 0);
    std::vector<int32_t> hTo, hW;
    hTo.reserve(m2);
    hW.reserve(m2);
    for (int64_t v = 0; v < n; ++v) {
        auto nb = ig.neighbors(v);
        hOffsets[v + 1] = hOffsets[v] + nb.size();
        for (uint32_t i = 0; i < nb.size(); ++i) { hTo.push_back(nb.to[i]); hW.push_back(nb.weight[i]); }
    }

    // Run-length beta tables (host_common.h): one per DISTINCT t0;
    // temperature replicas share via chainRunOff.
    const int64_t nRuns = static_cast<int64_t>(beta.runEnd.size());
    const int64_t nTables = beta.nTables;

    // Transfers/allocations are timed only to be REPORTED as excluded: the
    // kernel bracket must stay comparable to the CPU's solver-only timing.
    cudaEvent_t upStart, upStop, dnStart, dnStop;
    CUDA_CHECK(cudaEventCreate(&upStart));
    CUDA_CHECK(cudaEventCreate(&upStop));
    CUDA_CHECK(cudaEventCreate(&dnStart));
    CUDA_CHECK(cudaEventCreate(&dnStop));
    CUDA_CHECK(cudaEventRecord(upStart));

    // Per-chain slices padded to a multiple of 8 elements so Blocked8's
    // vector loads stay in bounds and aligned in every slice.  Padding is
    // zeroed, never scored, never downloaded.
    const int64_t nPad = (n + 7) & ~int64_t{7};

    DevBufs d;
    CUDA_CHECK(cudaMalloc(&d.offsets, (n + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.edgeTo, std::max<int64_t>(m2, 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.edgeWeight, std::max<int64_t>(m2, 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.spin, C * nPad * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d.bestSpin, C * nPad * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d.S, C * nPad * sizeof(SType)));
    CUDA_CHECK(cudaMemset(d.spin, 0, C * nPad * sizeof(int8_t)));
    CUDA_CHECK(cudaMemset(d.bestSpin, 0, C * nPad * sizeof(int8_t)));
    CUDA_CHECK(cudaMemset(d.S, 0, C * nPad * sizeof(SType)));
    CUDA_CHECK(cudaMalloc(&d.st, C * sizeof(AnnealState)));
    // nRuns >= nTables >= 1 always.
    CUDA_CHECK(cudaMalloc(&d.betaRunEnd, nRuns * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d.betaRunVal, nRuns * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.chainRunOff, C * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d.solverSeeds, C * sizeof(uint64_t)));
    // All -1 ("no winner here"); the kernel publishes and retracts entries
    // per sweep.
    CUDA_CHECK(cudaMalloc(&d.winnerGroup, C * nPad * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d.winnerGroup, 0xFF, C * nPad * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.phaseP, phases.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d.phaseIters, iters.size() * sizeof(uint64_t)));

    std::vector<uint64_t> hPhaseP(phases.begin(), phases.end());
    CUDA_CHECK(cudaMemcpy(d.betaRunEnd, beta.runEnd.data(), nRuns * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.betaRunVal, beta.runVal.data(), nRuns * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.chainRunOff, beta.chainRunOff.data(), C * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.solverSeeds, solverSeeds.data(), C * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.phaseP, hPhaseP.data(), phases.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.phaseIters, iters.data(), iters.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d.offsets, hOffsets.data(), (n + 1) * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    if (m2 > 0) {
        CUDA_CHECK(cudaMemcpy(d.edgeTo, hTo.data(), m2 * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.edgeWeight, hW.data(), m2 * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
    }

    // All chains' initial spins staged into one C x nPad image, one upload
    // each for spin and bestSpin.
    std::vector<int8_t> hSpin(static_cast<size_t>(C * nPad), 0);
    for (int64_t c = 0; c < C; ++c)
        std::copy(initialSpins[c].begin(), initialSpins[c].end(),
                  hSpin.begin() + static_cast<size_t>(c * nPad));
    CUDA_CHECK(cudaMemcpy(d.spin, hSpin.data(), C * nPad * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.bestSpin, hSpin.data(), C * nPad * sizeof(int8_t),
                          cudaMemcpyHostToDevice));

    // Each chain's currentVal/bestVal seeded from ITS initial cut, as serial
    // does.
    std::vector<AnnealState> hSt(static_cast<size_t>(C));
    for (int64_t c = 0; c < C; ++c) {
        hSt[c].currentVal = anneallib::solver::cutValueCsr(ig, initialSpins[c]);
        hSt[c].bestVal = hSt[c].currentVal;
        hSt[c].improved = 0;
        hSt[c].pad = 0;
    }
    CUDA_CHECK(cudaMemcpy(d.st, hSt.data(), C * sizeof(AnnealState),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(upStop));
    CUDA_CHECK(cudaEventSynchronize(upStop));

    DevCsr dg{d.offsets, d.edgeTo, d.edgeWeight, n};

    // >>> Everything above (alloc + H2D) is OUTSIDE the kernel bracket. <<<
    runMC<SType>(d, dg, nPad, C, static_cast<int64_t>(phases.size()), packed,
                 cfg.corr, maxP, tpb, kernelMs);

    // >>> Everything below (D2H) is OUTSIDE the kernel bracket. <<<
    CUDA_CHECK(cudaEventRecord(dnStart));

    std::vector<int8_t> hEnd(static_cast<size_t>(C * nPad));
    std::vector<int8_t> hBest(static_cast<size_t>(C * nPad));
    CUDA_CHECK(cudaMemcpy(hEnd.data(), d.spin, C * nPad * sizeof(int8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBest.data(), d.bestSpin, C * nPad * sizeof(int8_t),
                          cudaMemcpyDeviceToHost));
    endSpins.resize(static_cast<size_t>(C));
    bestSpins.resize(static_cast<size_t>(C));
    for (int64_t c = 0; c < C; ++c) {
        const auto off = static_cast<size_t>(c * nPad);
        endSpins[c].assign(hEnd.begin() + off, hEnd.begin() + off + n);
        bestSpins[c].assign(hBest.begin() + off, hBest.begin() + off + n);
    }

    std::vector<AnnealState> hOut(static_cast<size_t>(C));
    CUDA_CHECK(cudaMemcpy(hOut.data(), d.st, C * sizeof(AnnealState),
                          cudaMemcpyDeviceToHost));
    if (kernelBestVals) {
        kernelBestVals->resize(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) (*kernelBestVals)[c] = hOut[c].bestVal;
    }
    if (kernelFinalVals) {
        kernelFinalVals->resize(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) (*kernelFinalVals)[c] = hOut[c].currentVal;
    }

    CUDA_CHECK(cudaEventRecord(dnStop));
    CUDA_CHECK(cudaEventSynchronize(dnStop));

    // Report what was excluded so "kernel-only" is auditable; NOT added to
    // time-sec.
    float upMs = 0.0f, dnMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&upMs, upStart, upStop));
    CUDA_CHECK(cudaEventElapsedTime(&dnMs, dnStart, dnStop));
    const double h2dMB = (static_cast<double>(
                              (n + 1) * 4 + m2 * 8 + C * nPad * 2 +
                              nRuns * 12 +         // beta runs: int64 end + int32 value
                              C * (24 + 8 + 8) +   // st + chainRunOff + solverSeeds
                              static_cast<int64_t>(phases.size() + iters.size()) * 8)) /
                         1e6;
    std::fprintf(stderr,
                 "excluded-from-time-sec: alloc+H2D %.3f ms (%.2f MB, incl. %lld beta "
                 "table(s), %lld run(s), %.2f MB) | D2H %.3f ms | timed kernel %.3f ms\n",
                 upMs, h2dMB, static_cast<long long>(nTables),
                 static_cast<long long>(nRuns),
                 static_cast<double>(nRuns) * 12.0 / 1e6,
                 dnMs, *kernelMs);

    CUDA_CHECK(cudaEventDestroy(upStart));
    CUDA_CHECK(cudaEventDestroy(upStop));
    CUDA_CHECK(cudaEventDestroy(dnStart));
    CUDA_CHECK(cudaEventDestroy(dnStop));
    freeAll(d);
}

}  // namespace

void annealGpu(const anneallib::AnnealCsr& ig,
               int64_t maxWeightedDeg,
               const std::vector<std::vector<int8_t>>& initialSpins,
               const std::vector<double>& t0s,
               const std::vector<size_t>& phases,
               const std::vector<uint64_t>& iters,
               const std::vector<uint64_t>& solverSeeds,
               const LaunchConfig& cfg,
               std::vector<std::vector<int8_t>>& endSpins,
               std::vector<std::vector<int8_t>>& bestSpins,
               float* kernelMs,
               std::vector<int64_t>* kernelBestVals,
               std::vector<int64_t>* kernelFinalVals) {
    const size_t C = t0s.size();
    if (C == 0 || solverSeeds.size() != C || initialSpins.size() != C) {
        std::fprintf(stderr,
                     "Error: annealGpu needs non-empty, equal-length t0s (%zu), "
                     "solverSeeds (%zu) and initialSpins (%zu).\n",
                     t0s.size(), solverSeeds.size(), initialSpins.size());
        std::exit(EXIT_FAILURE);
    }
    if (ig.numVertices() == 0 || phases.empty()) {
        endSpins = initialSpins;
        bestSpins = initialSpins;
        *kernelMs = 0.0f;
        if (kernelBestVals) kernelBestVals->resize(C);
        if (kernelFinalVals) kernelFinalVals->resize(C);
        for (size_t c = 0; c < C; ++c) {
            const int64_t v = anneallib::solver::cutValueCsr(ig, initialSpins[c]);
            if (kernelBestVals) (*kernelBestVals)[c] = v;
            if (kernelFinalVals) (*kernelFinalVals)[c] = v;
        }
        return;
    }

    // Built before any device work, so an over-cap request fails loudly with
    // zero GPU state.
    const BetaRunTables beta = buildBetaRunTables(t0s, annealTotalSteps(iters));

    CUDA_CHECK(cudaSetDevice(cfg.device));

    if (cfg.sWidth == 64)
        runDispatch<int64_t>(ig, maxWeightedDeg, initialSpins, t0s, phases, iters,
                             solverSeeds, beta, cfg, endSpins, bestSpins, kernelMs,
                             kernelBestVals, kernelFinalVals);
    else
        runDispatch<int32_t>(ig, maxWeightedDeg, initialSpins, t0s, phases, iters,
                             solverSeeds, beta, cfg, endSpins, bestSpins, kernelMs,
                             kernelBestVals, kernelFinalVals);
}

}  // namespace gpu3
