#pragma once

#include <cstdint>

namespace anneallib {

// xoshiro256++ PRNG seeded via splitmix64.  This draws the INITIAL SPINS only,
// verbatim from the CPU reference so chain i starts from the same partition a
// CPU run with --seed seeds[i] would.  The annealing itself never touches it:
// the solver's randomness is the counter-based Philox stream in philox.cuh.
class Rng {
public:
    explicit Rng(uint64_t seed) { seedState(seed); }

    // n (1..32) random bits from a 64-bit reservoir; leftover bits that cannot
    // satisfy the request are discarded on refill.
    uint32_t bits(int n) {
        if (bitCount_ < n) {
            bitBuf_ = next();
            bitCount_ = 64;
        }
        uint32_t r = static_cast<uint32_t>(bitBuf_ & ((static_cast<uint64_t>(1) << n) - 1));
        bitBuf_ >>= n;
        bitCount_ -= n;
        return r;
    }

private:
    uint64_t s_[4];
    uint64_t bitBuf_ = 0;
    int bitCount_ = 0;

    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    uint64_t next() {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    void seedState(uint64_t seed) {
        // splitmix64 expansion of the seed into the 256-bit state.
        uint64_t z = seed;
        for (uint64_t& w : s_) {
            z += 0x9e3779b97f4a7c15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            w = x ^ (x >> 31);
        }
    }
};

} // namespace anneallib
