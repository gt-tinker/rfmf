#ifndef PHILOX_CUH
#define PHILOX_CUH

// Counter-based Gumbel sampling.  Philox4x32-10 has no state: the Gumbel
// handed to vertex v in sweep k is a pure function of (seed, k, v) --
// independent of launch geometry, vertex ordering, and kernel variant -- which
// is what makes chains isolated and the output reproducible (README section 8).
// This reproduces the CPU reference's SEMANTICS (identical LUT, bit layout,
// Q20 math), not its stateful xoshiro stream.

#include <cstdint>

// Usable from plain C++ TUs as well as nvcc TUs, so host and device share
// this code.
#if defined(__CUDACC__)
  #define PHILOX_HD __host__ __device__ __forceinline__
#else
  #define PHILOX_HD inline
#endif

namespace gpu3 {

// Q20: every Gumbel, beta, and score is in units of 2^-20.
constexpr int kGumbelFracBits = 20;

// coarse[i] = round(-log(-log((2i+1)/32)) * 2^20): a 16-point equiprobable
// quantile table.  The 7-bit fine offset below is a dither, NOT an
// interpolation -- substituting a "proper" inverse-CDF Gumbel would silently
// change the algorithm.  Spelled once as a macro because the device and the
// host each need their own copy.
#define GUMBEL_COARSE_INIT {                                          \
    -1303301, -903532, -648633, -438929, -249398,  -68827,  109563,   \
      290966,  480289,  683080,  906615, 1161749, 1466888, 1858654,   \
     2430921, 3617486 }

inline constexpr int32_t kGumbelCoarseHost[16] = GUMBEL_COARSE_INIT;
#if defined(__CUDACC__)
// Ordinary __device__ global, not __constant__: the persistent
// one-block-per-chain kernel hits this 64 B table from every thread every
// sweep, so it wants the L1 home rather than the constant cache.
__device__ static const int32_t kGumbelCoarseDevG[16] = GUMBEL_COARSE_INIT;
#endif

// kGumbelCoarse[15] + 127: the |gumbel| term of the int32-score overflow gate
// (annealScoreSafeInt32 in host_common.h).
constexpr int64_t kGumbelMaxMag = 3617613;
static_assert(kGumbelMaxMag == kGumbelCoarseHost[15] + 127,
              "Gumbel magnitude bound out of sync with the LUT");

// Low 4 bits index the coarse quantile, next 7 bits are the fine offset.
// `r` MUST be exactly 11 bits -- no internal mask (same as the CPU original).
PHILOX_HD int32_t gumbelFromBitsG(uint32_t r) {
#if defined(__CUDA_ARCH__)
    return kGumbelCoarseDevG[r & 0xF] + static_cast<int32_t>(r >> 4);
#else
    return kGumbelCoarseHost[r & 0xF] + static_cast<int32_t>(r >> 4);
#endif
}

// Philox4x32-10 (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3").
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // golden ratio
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // sqrt(3)-1

struct Philox4 { uint32_t v[4]; };
struct Philox2 { uint32_t v[2]; };

PHILOX_HD uint32_t philoxMulhi(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

PHILOX_HD void philoxRound(Philox4& ctr, const Philox2& key) {
    const uint32_t hi0 = philoxMulhi(kPhiloxM0, ctr.v[0]);
    const uint32_t lo0 = kPhiloxM0 * ctr.v[0];
    const uint32_t hi1 = philoxMulhi(kPhiloxM1, ctr.v[2]);
    const uint32_t lo1 = kPhiloxM1 * ctr.v[2];
    ctr.v[0] = hi1 ^ ctr.v[1] ^ key.v[0];
    ctr.v[1] = lo1;
    ctr.v[2] = hi0 ^ ctr.v[3] ^ key.v[1];
    ctr.v[3] = lo0;
}

// 10 rounds, 9 key bumps -- the standard Philox4x32-10 schedule.
PHILOX_HD Philox4 philox4x32_10(Philox4 ctr, Philox2 key) {
    // No #pragma unroll: the host compiler warns on it; nvcc unrolls anyway.
    philoxRound(ctr, key);
    for (int i = 0; i < 9; ++i) {
        key.v[0] += kPhiloxW0;
        key.v[1] += kPhiloxW1;
        philoxRound(ctr, key);
    }
    return ctr;
}

// The Gumbel stream: key = solverSeed (= --seed + kSolverSeedOffset),
// counter = (sweepId_lo, sweepId_hi, vertexBlock, 0), vertexBlock = v >> 3.
// One 128-bit Philox output serves 8 consecutive vertices: lane j = v & 7
// takes the j-th 16-bit field's low 11 bits.

PHILOX_HD Philox2 philoxKeyFromSeed(uint64_t solverSeed) {
    Philox2 k;
    k.v[0] = static_cast<uint32_t>(solverSeed & 0xFFFFFFFFull);
    k.v[1] = static_cast<uint32_t>(solverSeed >> 32);
    return k;
}

PHILOX_HD Philox4 gumbelBlockCounter(uint64_t sweepId, uint64_t vertexBlock) {
    Philox4 c;
    c.v[0] = static_cast<uint32_t>(sweepId & 0xFFFFFFFFull);
    c.v[1] = static_cast<uint32_t>(sweepId >> 32);
    c.v[2] = static_cast<uint32_t>(vertexBlock & 0xFFFFFFFFull);
    c.v[3] = static_cast<uint32_t>(vertexBlock >> 32);
    return c;
}

// Extract lane j's 11-bit field from a Philox block.  j in [0, 8).
PHILOX_HD uint32_t gumbelBitsFromBlock(const Philox4& out, uint32_t j) {
    const uint32_t word  = out.v[j >> 1];
    const uint32_t field = (j & 1u) ? (word >> 16) : (word & 0xFFFFu);
    return field & 0x7FFu;   // exactly 11 bits, as gumbelFromBits requires
}

// Packed per-group argmax key: [ biased score : 32 | ~id : 32 ].
// Biased score = (uint32)score ^ 0x80000000 maps int32 order onto uint32
// order; ~id makes the SMALLER id compare greater, reproducing candBeats'
// tie-break; empty slot = 0 (real candidates have ~id >= 0x80000000, so they
// can never tie it).  atomicMax over this key is associative and commutative,
// so the winner is independent of atomic arrival order.
//
// TWO SILENT TRAPS:
//   1. MUST be reduced with atomicMax(unsigned long long*, ...) -- the SIGNED
//      overload inverts the order for non-negative scores and surfaces only
//      as mysteriously bad quality.
//   2. Legal ONLY when annealScoreSafeInt32 holds; otherwise the kernel takes
//      the two-pass int64 path (see anneal_kernels.cu).

PHILOX_HD unsigned long long packScoreId(int32_t score, int32_t id) {
    const uint32_t biased = static_cast<uint32_t>(score) ^ 0x80000000u;
    return (static_cast<unsigned long long>(biased) << 32) |
           static_cast<unsigned long long>(~static_cast<uint32_t>(id));
}

PHILOX_HD int32_t unpackId(unsigned long long packed) {
    return static_cast<int32_t>(~static_cast<uint32_t>(packed & 0xFFFFFFFFull));
}

constexpr unsigned long long kPackedEmpty = 0ull;

}  // namespace gpu3

#endif  // PHILOX_CUH
