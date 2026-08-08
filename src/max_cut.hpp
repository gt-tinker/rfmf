#ifndef MAX_CUT_HPP
#define MAX_CUT_HPP

#include "immutable_graph.hpp"
#include <vector>
#include <cstdint>

namespace anneallib::solver {
    // Cut value of a dense +/-1 spin vector over a CSR snapshot; spin is
    // indexed by dense id, each undirected edge counted once.
    //
    // This is the artifact's INDEPENDENT check on the kernel: the solver
    // tracks its cut incrementally on the device, and every value printed on
    // stdout is recomputed here from the returned spins instead.  A corr[] or
    // double-counting bug shows up as a disagreement between the two.
    int64_t cutValueCsr(const AnnealCsr& ig, const std::vector<int8_t>& spin);
}

#endif // MAX_CUT_HPP
