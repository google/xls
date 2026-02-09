#include "xls/eco/gxl_parser.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <set>
#include <unistd.h>
#include <unordered_map>

#include "absl/log/log.h"
static XLSNode parse_node(tinyxml2::XMLElement *NodeElem);
static std::tuple<std::string, std::string, std::string, int>
parse_edge(tinyxml2::XMLElement *EdgeElem);
static tinyxml2::XMLElement *validate_and_load_gxl(const std::string &filename,
                                                   tinyxml2::XMLDocument &doc);
static void parse_nodes(XLSGraph &graph, tinyxml2::XMLElement *graph_elem);
static void parse_edges(XLSGraph &graph, tinyxml2::XMLElement *graph_elem,
                        const std::string &filename);
static void parse_graph_attrs(XLSGraph &graph, tinyxml2::XMLElement *graph_elem);
XLSGraph parse_gxl(const std::string &filename)
{
  XLSGraph graph;
  tinyxml2::XMLDocument doc;
  tinyxml2::XMLElement *graph_elem = validate_and_load_gxl(filename, doc);
  parse_graph_attrs(graph, graph_elem);
  parse_nodes(graph, graph_elem);
  parse_edges(graph, graph_elem, filename);
  // After parsing, compute labels and signatures
  graph.populate_node_signatures();
  graph.RefreshReturnAndIndex();
  return graph;
}
static void parse_graph_attrs(XLSGraph &graph, tinyxml2::XMLElement *graph_elem)
{
  for (tinyxml2::XMLElement *attr_elem = graph_elem->FirstChildElement("attr");
       attr_elem; attr_elem = attr_elem->NextSiblingElement("attr"))
  {
    const char *attr_name = attr_elem->Attribute("name");
    if (!attr_name)
      continue;
    if (strcmp(attr_name, "ret") == 0)
    {
      tinyxml2::XMLElement *string_elem =
          attr_elem->FirstChildElement("string");
      if (string_elem && string_elem->GetText())
      {
        graph.return_node_name = std::string(string_elem->GetText());
      }
    }
  }
}
static void parse_nodes(XLSGraph &graph, tinyxml2::XMLElement *graph_elem)
{
  for (tinyxml2::XMLElement *node_elem = graph_elem->FirstChildElement("node");
       node_elem; node_elem = node_elem->NextSiblingElement("node"))
  {
    auto node = parse_node(node_elem);
    graph.add_node(node);
  }
  VLOG(1) << "Parsed " << graph.nodes.size() << " nodes";
}
static void parse_edges(XLSGraph &graph, tinyxml2::XMLElement *graph_elem,
                        const std::string &filename)
{
  for (tinyxml2::XMLElement *edge_elem = graph_elem->FirstChildElement("edge");
       edge_elem; edge_elem = edge_elem->NextSiblingElement("edge"))
  {
    auto [from_node, to_node, cost_attrs, edge_index] = parse_edge(edge_elem);
    auto source_it = std::find_if(
        graph.nodes.begin(), graph.nodes.end(),
        [&from_node](const XLSNode &node)
        { return node.name == from_node; });
    auto target_it = std::find_if(
        graph.nodes.begin(), graph.nodes.end(),
        [&to_node](const XLSNode &node)
        { return node.name == to_node; });
    int source = (source_it != graph.nodes.end())
                     ? std::distance(graph.nodes.begin(), source_it)
                     : -1;
    int sink = (target_it != graph.nodes.end())
                   ? std::distance(graph.nodes.begin(), target_it)
                   : -1;
    if (source == -1 || sink == -1)
    {
      LOG(FATAL) << "Edge references non-existent node(s) in file '" << filename
          << "': from=\"" << from_node << "\" to=\"" << to_node << "\"";
    }
    graph.add_edge(XLSEdge(source, sink, cost_attrs, edge_index));
  }
  std::sort(graph.edges.begin(), graph.edges.end(),
            [](const XLSEdge &a, const XLSEdge &b)
            {
              if (a.endpoints.first != b.endpoints.first)
                return a.endpoints.first < b.endpoints.first;
              if (a.endpoints.second != b.endpoints.second)
                return a.endpoints.second < b.endpoints.second;
              return a.index < b.index;
            });
  graph.node_edges.clear();
  for (size_t i = 0; i < graph.edges.size(); ++i)
  {
    const auto &edge = graph.edges[i];
    graph.node_edges[edge.endpoints.first].push_back(i);
    graph.node_edges[edge.endpoints.second].push_back(i);
  }
  for (auto &[node_idx, edge_indices] : graph.node_edges)
  {
    std::sort(edge_indices.begin(), edge_indices.end(),
              [&graph](int a, int b)
              {
                return graph.edges[a].index < graph.edges[b].index;
              });
  }
  VLOG(1) << "Parsed " << graph.edges.size() << " edges";
}
static XLSNode parse_node(tinyxml2::XMLElement *NodeElem)
{
  const char *NodeId = NodeElem->Attribute("id");
  if (!NodeId)
  {
    LOG(FATAL) << "Node element has no id attribute";
  }
  std::string CostAttributes = "";
  std::vector<std::pair<std::string, std::string>> AllAttributes;
  for (tinyxml2::XMLElement *AttrElem = NodeElem->FirstChildElement("attr");
       AttrElem; AttrElem = AttrElem->NextSiblingElement("attr"))
  {
    const char *AttrName = AttrElem->Attribute("name");
    if (!AttrName)
      continue;
    tinyxml2::XMLElement *StringElem = AttrElem->FirstChildElement("string");
    if (StringElem && StringElem->GetText())
    {
      std::string AttrValue = StringElem->GetText();
      AllAttributes.push_back({AttrName, AttrValue});
      if (strcmp(AttrName, "cost_attributes") == 0)
      {
        CostAttributes = AttrValue;
      }
    }
  }
  XLSNode Node(NodeId, CostAttributes);
  Node.all_attributes = AllAttributes;
  VLOG(2) << "Parsed node: " << Node.name;
  VLOG(2) << "Cost attributes: " << Node.cost_attributes;
  VLOG(3) << "All attributes: " << Node.all_attributes.size();
  for (const auto &attr : Node.all_attributes)
  {
    VLOG(3) << "  " << attr.first << " = " << attr.second;
  }
  return Node;
}
static std::tuple<std::string, std::string, std::string, int>
parse_edge(tinyxml2::XMLElement *EdgeElem)
{
  const char *FromNode = EdgeElem->Attribute("from");
  const char *ToNode = EdgeElem->Attribute("to");
  if (!FromNode || !ToNode)
  {
    return std::make_tuple("", "", "", 0);
  }
  std::string CostAttributes = "";
  std::optional<int> Index = std::nullopt;
  for (tinyxml2::XMLElement *AttrElem = EdgeElem->FirstChildElement("attr");
       AttrElem; AttrElem = AttrElem->NextSiblingElement("attr"))
  {
    const char *AttrName = AttrElem->Attribute("name");
    if (!AttrName)
      continue;
    tinyxml2::XMLElement *StringElem = AttrElem->FirstChildElement("string");
    if (!StringElem)
      continue;
    const char *AttrValue = StringElem->GetText();
    if (!AttrValue)
      continue;
    if (strcmp(AttrName, "cost_attributes") == 0)
    {
      CostAttributes = AttrValue;
      size_t idx_pos = CostAttributes.find("|index=");
      if (!Index.has_value() && idx_pos != std::string::npos)
      {
        size_t start = idx_pos + 7;
        size_t end = CostAttributes.find("|", start);
        if (end == std::string::npos)
          end = CostAttributes.length();
        try
        {
          Index = std::stoi(CostAttributes.substr(start, end - start));
        }
        catch (...)
        {
        }
      }
    }
    else if (strcmp(AttrName, "index") == 0)
    {
      try
      {
        Index = std::stoi(AttrValue);
      }
      catch (...)
      {
      }
    }
  }
  int EdgeIndex = Index.has_value() ? Index.value() : 0;
  VLOG(2) << "Parsed edge: " << FromNode << " -> " << ToNode;
  if (!CostAttributes.empty())
  {
    VLOG(2) << "Cost attributes: " << CostAttributes;
  }
  VLOG(2) << "Edge index: " << EdgeIndex;
  return std::make_tuple(FromNode, ToNode, CostAttributes, EdgeIndex);
}
static tinyxml2::XMLElement *validate_and_load_gxl(const std::string &filename,
                                                   tinyxml2::XMLDocument &doc)
{
  {
    std::ifstream in(filename);
    if (!in.good())
    {
      char cwd[4096];
      const char *cwd_ptr =
          getcwd(cwd, sizeof(cwd)) ? cwd : "(cwd unavailable)";
      LOG(FATAL) << "Failed to open GXL file: " << filename << " (cwd=" << cwd_ptr
          << ")";
    }
  }
  const auto load_status = doc.LoadFile(filename.c_str());
  if (load_status != tinyxml2::XML_SUCCESS)
  {
    LOG(FATAL) << "Failed to load GXL file: " << filename
        << " | TinyXML2 ErrorID=" << doc.ErrorID() << " ("
        << (doc.ErrorName() ? doc.ErrorName() : "?") << ")"
        << ", Msg: " << (doc.ErrorStr() ? doc.ErrorStr() : "<no message>");
  }
  tinyxml2::XMLElement *root = doc.RootElement();
  if (!root || strcmp(root->Name(), "gxl") != 0)
  {
    LOG(FATAL) << "Invalid GXL file: root element is not 'gxl' (found '"
        << (root ? root->Name() : "<null>") << "')";
  }
  tinyxml2::XMLElement *graph_elem = root->FirstChildElement("graph");
  if (!graph_elem)
  {
    LOG(FATAL) << "No <graph> element found under <gxl> in file '" << filename << "'";
    exit(1);
  }
  return graph_elem;
}
