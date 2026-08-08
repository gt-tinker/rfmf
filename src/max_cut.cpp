#include "max_cut.hpp"

namespace anneallib::solver {

int64_t cutValueCsr(const AnnealCsr& ig, const std::vector<int8_t>& spin) {
    int64_t value = 0;
    const uint64_t n = ig.numVertices();
    for (uint64_t v = 0; v < n; ++v) {
        auto nb = ig.neighbors(static_cast<int64_t>(v));
        for (uint32_t i = 0; i < nb.size(); ++i)
            if (static_cast<uint64_t>(nb.to[i]) > v && spin[v] != spin[nb.to[i]])
                value += nb.weight[i];
    }
    return value;
}

} // namespace anneallib::solver
