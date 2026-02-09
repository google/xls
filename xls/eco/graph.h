#ifndef GRAPH_H
#define GRAPH_H

#include <cstddef>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct XLSNode
{
  std::string name;
  std::string cost_attributes;
  bool pinned = false;
  int index = -1; // current index in graph.nodes
  std::optional<int> mcs_map_index; // index in MCS mapping, if mapped
  std::vector<std::pair<std::string, std::string>> all_attributes;
  // label: hash of cost_attributes (node-local label)
  std::size_t label = 0;
  // signature: hash(label, ordered(incoming labels), unordered(outgoing labels))
  std::size_t signature = 0;
  std::vector<std::size_t> incoming_labels;
  std::vector<std::size_t> outgoing_labels;
  XLSNode(const std::string &node_name, const std::string &cost_attrs = "");
};

struct XLSEdge
{
  std::pair<int, int> endpoints; // source, sink
  std::string cost_attributes;
  int index;
  // label: hash of cost_attributes (edge-local label)
  std::size_t label = 0;
  XLSEdge(int source, int sink,
          const std::string &cost_attrs = "", int idx = 0);
};

// Custom hash function for std::pair<int, int>
struct PairHash
{
  std::size_t operator()(const std::pair<int, int> &p) const noexcept
  {
    return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
  }
};

struct XLSGraph
{
  std::vector<XLSNode> nodes;
  std::vector<XLSEdge> edges;
  std::unordered_map<int, std::vector<int>> node_edges;
  std::unordered_map<std::pair<int, int>, int, PairHash> edge_counts;
  std::unordered_map<std::string, int> node_name_to_index;
  std::optional<std::string> return_node_name;
  
  // Mapping between current indices and original indices after cutting
  std::vector<int> original_indices;  // current_index -> original_index
  std::unordered_map<int, int> current_indices;  // original_index -> current_index

  XLSGraph();
  XLSGraph(const XLSGraph& other) = default;  // Copy constructor
  XLSGraph& operator=(const XLSGraph& other) = default;  // Copy assignment
  bool has_edge(int u, int v, int index) const;
  int count_edges(int u, int v) const;
  // Convenience: presence-only check
  bool has_edge(int u, int v) const;
  int add_node(const XLSNode &Node);
  int add_edge(const XLSEdge &Edge);
  std::vector<const XLSEdge *> get_node_edges(int node_index) const;
  std::vector<int> get_edges_between(int u, int v) const;
  std::vector<int> get_outgoing_neighbors(int node_index) const;
  std::vector<int> get_incoming_neighbors(int node_index) const;
  // return incoming + outgoing edges in a single vector
  std::vector<int> get_neighbors(int node_index) const;
  // compute node labels (from cost_attributes) and signatures
  void populate_node_signatures();
  // pin nodes (set pinned=true for specified node indices)
  void PinNodes(const std::vector<int> &node_indices);
  // cut nodes: remove specified nodes and their connected edges from the graph
  void Cut(const std::vector<int> &node_indices);
  void RefreshAdjacency();
  void RefreshEdgeCounts();
  void RefreshReturnAndIndex();
  // validate and clean up edges with missing endpoints
  void ValidateEdges();
};

#endif
