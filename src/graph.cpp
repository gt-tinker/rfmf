#include "graph.hpp"
#include <stdexcept>

namespace anneallib {

void Graph::addVertex(int64_t vertexId) {
    vertices.insert(vertexId);
    adj.emplace(vertexId, std::unordered_map<int64_t, int64_t>{});
}

void Graph::addEdge(int64_t u, int64_t v, int64_t weight) {
    addVertex(u);
    addVertex(v);
    if (adj[u].count(v)) return; // duplicate edge
    adj[u][v] = weight;
    adj[v][u] = weight;
}

void Graph::removeVertex(int64_t vertexId) {
    auto it = adj.find(vertexId);
    if (it == adj.end()) return;
    for (const auto& [neighbor, weight] : it->second) {
        adj[neighbor].erase(vertexId);
    }
    adj.erase(it);
    vertices.erase(vertexId);
}

uint64_t Graph::degree(int64_t vertexId) const {
    auto it = adj.find(vertexId);
    if (it == adj.end()) throw std::runtime_error("Vertex not found");
    return it->second.size();
}

uint64_t Graph::numVertices() const {
    return vertices.size();
}

const std::unordered_set<int64_t>& Graph::getVertices() const {
    return vertices;
}

const std::unordered_map<int64_t, int64_t>& Graph::getEdges(int64_t vertexId) const {
    auto it = adj.find(vertexId);
    if (it == adj.end()) throw std::runtime_error("Vertex not found");
    return it->second;
}

} // namespace anneallib
