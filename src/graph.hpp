#ifndef GRAPH_HPP
#define GRAPH_HPP

// Mutable undirected integer-weighted graph, adjacency-map based.  Used only
// on the load path (file => Graph => AnnealCsr) and freed before the kernel
// launches; the annealing hot loop reads the CSR snapshot instead.

#include <unordered_map>
#include <unordered_set>
#include <cstdint>

namespace anneallib {
    class Graph {
    private:
        std::unordered_set<int64_t> vertices;
        // vertex -> (neighbor -> weight)
        std::unordered_map<int64_t, std::unordered_map<int64_t, int64_t>> adj;

    public:
        Graph() = default;

        // No-op if vertexId already exists.
        void addVertex(int64_t vertexId);

        // Adds missing vertices; no-op if the edge exists (keeps original
        // weight), which is how the loader dedups a repeated edge line.
        void addEdge(int64_t u, int64_t v, int64_t weight);

        // Removes the vertex and all incident edges.  No-op if absent.
        void removeVertex(int64_t vertexId);

        uint64_t degree(int64_t vertexId) const;
        uint64_t numVertices() const;
        const std::unordered_set<int64_t>& getVertices() const;

        // Adjacency map of vertexId; throws if absent.
        const std::unordered_map<int64_t, int64_t>& getEdges(int64_t vertexId) const;

        ~Graph() = default;
    };
}


#endif // GRAPH_HPP
