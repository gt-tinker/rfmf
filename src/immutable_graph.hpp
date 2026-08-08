#ifndef IMMUTABLE_GRAPH_HPP
#define IMMUTABLE_GRAPH_HPP

// Read-only CSR snapshot of a Graph, frozen before the launch and read (never
// written) by every chain.  Vertices are renumbered to dense ids [0, n) in
// ascending original-id order -- this is the id space the printed partition
// bitstrings are indexed by, NOT the original Gset numbering.  Storage is
// structure-of-arrays (edgeTo_ / edgeWeight_) so neighbour walks stay
// contiguous and vectorizable.  IdT/WeightT/OffsetT are template parameters so
// the two 2*numEdges arrays, which dominate memory bandwidth, can be sized to
// the known input range; the solver instantiates AnnealCsr (below).

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#include "graph.hpp"

namespace anneallib {

    template<class IdT     = std::int32_t,
             class WeightT = std::int32_t,
             class OffsetT = std::uint32_t>
    class ImmutableGraph {
    public:
        using id_type     = IdT;
        using weight_type = WeightT;
        using offset_type = OffsetT;

    private:
        // CSR: vertex v's neighbours are edgeTo_[offsets_[v] .. offsets_[v+1])
        // with matching weights in edgeWeight_.  Each undirected edge is
        // stored twice.
        std::vector<OffsetT>  offsets_;     // size n + 1
        std::vector<IdT>      edgeTo_;      // size 2 * numEdges
        std::vector<WeightT>  edgeWeight_;  // size 2 * numEdges

        std::vector<std::int64_t> newToOld_;  // dense new id -> original id

        // CSR builder.  `oldToNew` must be a bijection from g's vertex set
        // onto [0, numVertices); it is consumed here and not retained.
        void build(const Graph& g,
                   const std::unordered_map<std::int64_t, std::int64_t>& oldToNew) {
            const std::uint64_t n = g.numVertices();

            if (oldToNew.size() != n)
                throw std::runtime_error("ImmutableGraph: mapping size != numVertices");

            newToOld_.assign(n, 0);
            std::vector<bool> seen(n, false);
            for (std::int64_t old : g.getVertices()) {
                auto it = oldToNew.find(old);
                if (it == oldToNew.end())
                    throw std::runtime_error("ImmutableGraph: vertex missing from mapping");
                std::int64_t nid = it->second;
                if (nid < 0 || static_cast<std::uint64_t>(nid) >= n)
                    throw std::runtime_error("ImmutableGraph: new id out of range");
                if (seen[nid])
                    throw std::runtime_error("ImmutableGraph: new id collision (not one-to-one)");
                seen[nid] = true;
                newToOld_[nid] = old;
            }

            // Pass 1: per-vertex degree -> offsets via prefix sum.
            offsets_.assign(n + 1, 0);
            for (std::uint64_t v = 0; v < n; ++v)
                offsets_[v + 1] = static_cast<OffsetT>(offsets_[v] + g.degree(newToOld_[v]));

            // Pass 2: fill the two parallel neighbour arrays with dense ids.
            edgeTo_.resize(offsets_[n]);
            edgeWeight_.resize(offsets_[n]);
            for (std::uint64_t v = 0; v < n; ++v) {
                OffsetT pos = offsets_[v];
                for (const auto& [nbOld, w] : g.getEdges(newToOld_[v])) {
                    edgeTo_[pos]     = static_cast<IdT>(oldToNew.at(nbOld));
                    edgeWeight_[pos] = static_cast<WeightT>(w);
                    ++pos;
                }
            }

            // Pass 3: sort each vertex's slice ascending by dense neighbour id.
            // NOT optional: the raw fill order above is unordered_map hash
            // order, which is not stable across standard libraries.  Sorting
            // fixes the summation order of every neighbour loop on the device,
            // and with it the bit-exactness of the results.
            std::vector<std::pair<IdT, WeightT>> slice;
            for (std::uint64_t v = 0; v < n; ++v) {
                const OffsetT lo = offsets_[v], hi = offsets_[v + 1];
                slice.clear();
                for (OffsetT i = lo; i < hi; ++i)
                    slice.emplace_back(edgeTo_[i], edgeWeight_[i]);
                std::sort(slice.begin(), slice.end(),
                          [](const std::pair<IdT, WeightT>& a,
                             const std::pair<IdT, WeightT>& b) { return a.first < b.first; });
                for (OffsetT i = lo; i < hi; ++i) {
                    edgeTo_[i]     = slice[i - lo].first;
                    edgeWeight_[i] = slice[i - lo].second;
                }
            }
        }

    public:
        // New ids ordered by ascending original id.
        explicit ImmutableGraph(const Graph& g) {
            std::vector<std::int64_t> ids(g.getVertices().begin(), g.getVertices().end());
            std::sort(ids.begin(), ids.end());

            std::unordered_map<std::int64_t, std::int64_t> oldToNew;
            oldToNew.reserve(ids.size());
            for (std::size_t i = 0; i < ids.size(); ++i)
                oldToNew[ids[i]] = static_cast<std::int64_t>(i);

            build(g, oldToNew);
        }

        // Empty graph (n = 0), so callers can hold one by value and fill later.
        ImmutableGraph() : offsets_(1, 0) {}

        // The user-declared destructor suppresses the implicit move members;
        // declare all five so the big arrays move rather than copy.
        ImmutableGraph(ImmutableGraph&&) noexcept = default;
        ImmutableGraph& operator=(ImmutableGraph&&) noexcept = default;
        ImmutableGraph(const ImmutableGraph&) = default;
        ImmutableGraph& operator=(const ImmutableGraph&) = default;

        std::uint64_t numVertices() const { return newToOld_.size(); }
        std::uint64_t numEdges() const { return edgeTo_.size() / 2; }  // each edge stored twice

        // Contiguous view of v's neighbours as two parallel same-indexed
        // arrays, valid only while this graph lives.  Throws on invalid v.
        struct NeighborSpan {
            const IdT*     to;
            const WeightT* weight;
            OffsetT        n;
            OffsetT size() const { return n; }
        };
        NeighborSpan neighbors(std::int64_t v) const {
            if (v < 0 || static_cast<std::uint64_t>(v) >= newToOld_.size())
                throw std::out_of_range("ImmutableGraph::neighbors: invalid vertex");
            return NeighborSpan{edgeTo_.data() + offsets_[v],
                                edgeWeight_.data() + offsets_[v],
                                static_cast<OffsetT>(offsets_[v + 1] - offsets_[v])};
        }

        ~ImmutableGraph() = default;
    };

    // CSR type the annealing kernels instantiate.  int32_t weights cover every
    // Gset instance (the reader in read_gset.cpp scales float-weighted inputs
    // to fit); a build for wider weights can override the type with
    // -DANNEAL_WEIGHT_T=std::int64_t, which is untested here.
#ifndef ANNEAL_WEIGHT_T
#define ANNEAL_WEIGHT_T std::int32_t
#endif
    using AnnealCsr = ImmutableGraph<std::int32_t, ANNEAL_WEIGHT_T, std::uint32_t>;
}

#endif // IMMUTABLE_GRAPH_HPP
