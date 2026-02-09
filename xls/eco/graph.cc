#include "xls/eco/graph.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <unordered_set>

#include "absl/log/log.h"

XLSNode::XLSNode(const std::string& node_name, const std::string& cost_attrs)
    : name(node_name),
      cost_attributes(cost_attrs),
      label(std::hash<std::string>{}(cost_attributes)) {}
XLSEdge::XLSEdge(int source, int sink, const std::string& cost_attrs, int idx)
    : endpoints(source, sink),
      cost_attributes(cost_attrs),
      index(idx),
      label(std::hash<std::string>{}(cost_attributes)) {}
XLSGraph::XLSGraph() {}

// Helper function to extract op value from cost_attributes string
static std::string extract_op(const std::string& cost_attributes) {
  size_t op_pos = cost_attributes.find("op=");
  if (op_pos == std::string::npos) return "";
  size_t start = op_pos + 3;
  size_t end = cost_attributes.find("|", start);
  if (end == std::string::npos) end = cost_attributes.length();
  return cost_attributes.substr(start, end - start);
}

int XLSGraph::add_node(const XLSNode& node) {
  int node_index = nodes.size();
  nodes.push_back(node);
  nodes.back().index = node_index;
  node_name_to_index[node.name] = node_index;
  VLOG(2) << "Added node: " << node.name << " at index " << node_index
          << " with cost attributes: " << node.cost_attributes;
  VLOG(3) << "Node Label: " << node.label;
  return node_index;
}

int XLSGraph::add_edge(const XLSEdge& edge) {
  auto key = std::make_pair(edge.endpoints.first, edge.endpoints.second);
  int& edge_count = edge_counts[key];  // use reference for direct increment

  // Disable node expansion - use original node indices directly
  int source_index = edge.endpoints.first;
  int sink_index = edge.endpoints.second;
  edge_count++;  // Still track count for potential future use

  // Assert that edge source and sink ops match the source and sink nodes ops
  std::string source_op = extract_op(nodes[source_index].cost_attributes);
  std::string sink_op = extract_op(nodes[sink_index].cost_attributes);

  // Extract source_op and sink_op from edge cost_attributes if present
  std::string edge_source_op;
  std::string edge_sink_op;

  size_t edge_source_op_pos = edge.cost_attributes.find("source_op=");
  if (edge_source_op_pos != std::string::npos) {
    size_t start = edge_source_op_pos + 10;
    size_t end = edge.cost_attributes.find("|", start);
    if (end == std::string::npos) end = edge.cost_attributes.length();
    edge_source_op = edge.cost_attributes.substr(start, end - start);
  }

  size_t edge_sink_op_pos = edge.cost_attributes.find("sink_op=");
  if (edge_sink_op_pos != std::string::npos) {
    size_t start = edge_sink_op_pos + 8;
    size_t end = edge.cost_attributes.find("|", start);
    if (end == std::string::npos) end = edge.cost_attributes.length();
    edge_sink_op = edge.cost_attributes.substr(start, end - start);
  }

  // If edge specifies source_op or sink_op, they must match the nodes' ops
  if (!edge_source_op.empty() && source_op != edge_source_op) {
    LOG(FATAL) << "Edge source_op mismatch: edge specifies '" << edge_source_op
               << "' but source node has op='" << source_op << "'";
  }
  if (!edge_sink_op.empty() && sink_op != edge_sink_op) {
    LOG(FATAL) << "Edge sink_op mismatch: edge specifies '" << edge_sink_op
        << "' but sink node has op='" << sink_op << "'";
  }

  edges.emplace_back(source_index, sink_index, edge.cost_attributes, edge.index);

  int edge_idx = static_cast<int>(edges.size()) - 1;
  node_edges[source_index].push_back(edge_idx);
  node_edges[sink_index].push_back(edge_idx);

  auto sort_by_index = [this](int a, int b) {
    return edges[a].index < edges[b].index;
  };
  std::sort(node_edges[source_index].begin(), node_edges[source_index].end(),
            sort_by_index);
  std::sort(node_edges[sink_index].begin(), node_edges[sink_index].end(),
            sort_by_index);
  VLOG(2) << "Added edge: " << nodes[source_index].name << " -> "
          << nodes[sink_index].name << " with index: " << edge.index;
  VLOG(3) << "  Source node attributes: "
          << nodes[source_index].cost_attributes;
  VLOG(3) << "  Sink node attributes: " << nodes[sink_index].cost_attributes;
  VLOG(3) << "  Edge attributes: " << edge.cost_attributes;
  VLOG(3) << "  Edge label: " << edge.label;
  return edge_idx;
}

std::vector<const XLSEdge*> XLSGraph::get_node_edges(int node_index) const {
  std::vector<const XLSEdge*> result;
  auto it = node_edges.find(node_index);
  if (it != node_edges.end()) {
    for (int edge_idx : it->second) {
      result.push_back(&edges[edge_idx]);
    }
  }
  return result;
}

std::vector<int> XLSGraph::get_edges_between(int u, int v) const {
  std::vector<int> result;
  auto it = node_edges.find(u);
  if (it == node_edges.end()) return result;

  for (int edge_idx : it->second) {
    const XLSEdge& edge = edges[edge_idx];
    if (edge.endpoints.first == u && edge.endpoints.second == v) {
      result.push_back(edge.index);
    }
  }
  return result;
}

std::vector<int> XLSGraph::get_outgoing_neighbors(int node_index) const {
  std::vector<int> result;
  auto it = node_edges.find(node_index);
  if (it != node_edges.end()) {
    for (int edge_idx : it->second) {
      const XLSEdge& edge = edges[edge_idx];
      if (edge.endpoints.first == node_index) {
        result.push_back(edge.endpoints.second);
      }
    }
  }
  return result;
}

std::vector<int> XLSGraph::get_incoming_neighbors(int node_index) const {
  std::vector<int> result;
  auto it = node_edges.find(node_index);
  if (it != node_edges.end()) {
    for (int edge_idx : it->second) {
      const XLSEdge& edge = edges[edge_idx];
      if (edge.endpoints.second == node_index) {
        result.push_back(edge.endpoints.first);
      }
    }
  }
  return result;
}

std::vector<int> XLSGraph::get_neighbors(int node_index) const {
  auto out = get_outgoing_neighbors(node_index);
  auto in = get_incoming_neighbors(node_index);
  std::vector<int> result;
  result.reserve(out.size() + in.size());
  result.insert(result.end(), out.begin(), out.end());
  result.insert(result.end(), in.begin(), in.end());
  return result;
}

bool XLSGraph::has_edge(int u, int v, int index /* = -1 */) const {
  auto it = node_edges.find(u);
  if (it == node_edges.end()) return false;

  for (int edge_idx : it->second) {
    const XLSEdge& e = edges[edge_idx];
    if (e.endpoints.first == u && e.endpoints.second == v) {
      // if index == -1, accept any; otherwise, require exact match
      if (index == -1 || e.index == index) return true;
    }
  }
  return false;
}
bool XLSGraph::has_edge(int u, int v) const { return has_edge(u, v, -1); }

int XLSGraph::count_edges(int u, int v) const {
  auto it = node_edges.find(u);
  if (it == node_edges.end()) return 0;

  int count = 0;
  for (int edge_idx : it->second) {
    const XLSEdge& e = edges[edge_idx];
    if (e.endpoints.first == u && e.endpoints.second == v) ++count;
  }
  return count;
}

static inline std::size_t hash_combine(std::size_t seed, std::size_t v) {
  seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) +
          (seed >> 2);  // The golden ratio
  return seed;
}

static inline std::size_t hash_vector(const std::vector<std::size_t>& vals) {
  std::size_t seed = 0;
  for (auto v : vals) seed = hash_combine(seed, v);
  return seed;
}

void XLSGraph::populate_node_signatures() {
  // clear old
  for (auto& n : nodes) {
    n.incoming_labels.clear();
    n.outgoing_labels.clear();
    n.signature = 0;
  }

  struct NeighborInfo {
    int edge_index;
    int neighbor_index;
    std::size_t label;
  };

  auto collect_incoming_ordered = [&](int node_index) {
    std::vector<NeighborInfo> ordered;
    auto it = node_edges.find(node_index);
    if (it != node_edges.end()) {
      for (int edge_idx : it->second) {
        const XLSEdge& edge = edges[edge_idx];
        if (edge.endpoints.second != node_index) continue;
        int neighbor = edge.endpoints.first;
        ordered.push_back({edge.index, neighbor, nodes[neighbor].label});
      }
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const NeighborInfo& a, const NeighborInfo& b) {
                if (a.edge_index != b.edge_index) {
                  return a.edge_index < b.edge_index;
                }
                return a.neighbor_index < b.neighbor_index;
              });
    std::vector<std::size_t> labels;
    labels.reserve(ordered.size());
    for (const NeighborInfo& info : ordered) {
      labels.push_back(info.label);
    }
    return labels;
  };

  auto collect_outgoing_unordered = [&](int node_index) {
    std::vector<std::size_t> labels;
    auto it = node_edges.find(node_index);
    if (it != node_edges.end()) {
      for (int edge_idx : it->second) {
        const XLSEdge& edge = edges[edge_idx];
        if (edge.endpoints.first != node_index) continue;
        int neighbor = edge.endpoints.second;
        labels.push_back(nodes[neighbor].label);
      }
    }
    std::sort(labels.begin(), labels.end());
    return labels;
  };

  // populate incoming/outgoing labels per node, preserving operand order
  for (std::size_t u = 0; u < nodes.size(); ++u) {
    nodes[u].incoming_labels = collect_incoming_ordered(static_cast<int>(u));
    nodes[u].outgoing_labels =
        collect_outgoing_unordered(static_cast<int>(u));

    // signature = hash(label, ordered(incoming_labels),
    // unordered(outgoing_labels))
    std::size_t sig = 0;
    sig = hash_combine(sig, nodes[u].label);
    sig = hash_combine(sig, hash_vector(nodes[u].incoming_labels));
    sig = hash_combine(sig, hash_vector(nodes[u].outgoing_labels));
    nodes[u].signature = sig;
  }

  VLOG(1) << "Populated node labels and signatures for " << nodes.size()
          << " nodes";
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].pinned) {
      VLOG(2) << "Pinned node: " << nodes[i].name
              << " attrs=" << nodes[i].cost_attributes;
    }
  }
}

void XLSGraph::PinNodes(const std::vector<int>& node_indices) {
  for (int idx : node_indices) {
    if (idx >= 0 && idx < static_cast<int>(nodes.size())) {
      nodes[idx].pinned = true;
      VLOG(2) << "Pinned node: " << nodes[idx].name << " (index " << idx << ")";
    } else {
      VLOG(1) << "Warning: Attempted to pin invalid node index: " << idx;
    }
  }
  VLOG(1) << "Pinned " << node_indices.size() << " nodes";
}

void XLSGraph::Cut(const std::vector<int>& node_indices) {
  if (node_indices.empty()) {
    VLOG(1) << "Cut: No nodes to remove";
    return;
  }

  // 1) Build removal set, EXCLUDING pinned nodes
  std::unordered_set<int> to_remove;
  to_remove.reserve(node_indices.size());
  for (int idx : node_indices) {
    if (idx >= 0 && idx < (int)nodes.size() && !nodes[idx].pinned)
      to_remove.insert(idx);
  }
  VLOG(1) << "Cut: Removing " << to_remove.size() << " nodes (pinned kept)";

  // Early exit if nothing to do
  if (to_remove.empty()) {
    VLOG(1) << "Cut: nothing to remove after excluding pinned";
    return;
  }

  // 2) Remove edges that touch ANY removed node (boundary edges are safe:
  //     their endpoints are not in to_remove)
  const size_t old_edge_count = edges.size();
  edges.erase(std::remove_if(edges.begin(), edges.end(),
                             [&](const XLSEdge& e) {
                               return to_remove.count(e.endpoints.first) ||
                                      to_remove.count(e.endpoints.second);
                             }),
              edges.end());
  VLOG(1) << "Cut: Removed " << (old_edge_count - edges.size())
          << " edges incident to removed nodes";

  // 3) Compact nodes (delete removed, keep others)
  std::vector<XLSNode> new_nodes;
  new_nodes.reserve(nodes.size() - to_remove.size());
  std::vector<int> old_to_new(nodes.size(), -1);

  for (int i = 0; i < (int)nodes.size(); ++i) {
    if (to_remove.count(i)) continue;
    old_to_new[i] = (int)new_nodes.size();
    new_nodes.emplace_back(std::move(nodes[i]));
  }
  nodes.swap(new_nodes);

  // 4) Remap edge endpoints (now guaranteed valid since we removed incident
  // edges)
  for (auto& e : edges) {
    e.endpoints.first = old_to_new[e.endpoints.first];
    e.endpoints.second = old_to_new[e.endpoints.second];
  }

  // 5) Refresh derived structures and validate (should remove ~0 now)
  RefreshAdjacency();
  RefreshEdgeCounts();
  RefreshReturnAndIndex();
  ValidateEdges();  // should be a no-op or very small now

  VLOG(1) << "Cut complete: " << nodes.size() << " nodes and " << edges.size()
          << " edges remaining.";
}

void XLSGraph::RefreshAdjacency() {
  node_edges.clear();
  node_edges.reserve(nodes.size());

  for (int i = 0; i < (int)nodes.size(); ++i) node_edges[i] = {};

  for (int e_idx = 0; e_idx < (int)edges.size(); ++e_idx) {
    const auto& e = edges[e_idx];
    int src = e.endpoints.first;
    int dst = e.endpoints.second;

    if (src >= 0 && src < (int)nodes.size()) node_edges[src].push_back(e_idx);
    if (dst >= 0 && dst < (int)nodes.size()) node_edges[dst].push_back(e_idx);
  }

  for (auto& [node_idx, edge_indices] : node_edges) {
    std::sort(edge_indices.begin(), edge_indices.end(), [this](int a, int b) {
      if (edges[a].index != edges[b].index) {
        return edges[a].index < edges[b].index;
      }
      if (edges[a].endpoints.first != edges[b].endpoints.first) {
        return edges[a].endpoints.first < edges[b].endpoints.first;
      }
      if (edges[a].endpoints.second != edges[b].endpoints.second) {
        return edges[a].endpoints.second < edges[b].endpoints.second;
      }
      return a < b;
    });
  }

  VLOG(2) << "RefreshAdjacency: rebuilt adjacency for " << node_edges.size()
          << " nodes.";
}
void XLSGraph::RefreshEdgeCounts() {
  edge_counts.clear();

  for (const auto& e : edges) {
    const auto& key = e.endpoints;
    edge_counts[key]++;
  }

  VLOG(2) << "RefreshEdgeCounts: counted " << edge_counts.size()
        << " unique edge pairs.";
}

void XLSGraph::RefreshReturnAndIndex() {
  node_name_to_index.clear();
  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    nodes[i].index = i;
    node_name_to_index[nodes[i].name] = i;
  }
  if (return_node_name.has_value()) {
    if (node_name_to_index.find(*return_node_name) ==
        node_name_to_index.end()) {
      return_node_name.reset();
    }
  }
}
void XLSGraph::ValidateEdges() {
  int before = (int)edges.size();

  edges.erase(std::remove_if(edges.begin(), edges.end(),
                             [&](const XLSEdge& e) {
                               int src = e.endpoints.first;
                               int dst = e.endpoints.second;
                               return src < 0 || dst < 0 ||
                                      src >= (int)nodes.size() ||
                                      dst >= (int)nodes.size();
                             }),
              edges.end());

  int removed = before - (int)edges.size();
  if (removed > 0)
    VLOG(1) << "ValidateEdges: removed " << removed << " invalid edges.";

  RefreshAdjacency();
  RefreshEdgeCounts();
}
