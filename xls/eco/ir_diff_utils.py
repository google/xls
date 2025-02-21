#
# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of utility classes and functions for the ECO tool."""

import os
import time
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import networkx as nx

from xls.eco import ir_diff


def visualize_ir(graph: nx.DiGraph) -> None:
  """Visualizes a directed graph using a top-down layout with horizontal node spreading.

  Args:
    graph: A NetworkX directed graph object.

  Returns:
    None (function generates a plot and displays it).
  """

  # Assign levels based on shortest paths from root nodes
  levels = _assign_levels_to_nodes(graph)
  # Calculate initial positions with top-down layout
  pos = {node: (0, -level) for node, level in levels.items()}
  # Spread nodes horizontally within each level
  pos = _spread_nodes_horizontally(pos)
  # Configure plot and node colors
  num_nodes = len(graph.nodes())
  num_edges = len(graph.edges())
  total_elements = num_nodes + num_edges
  scale_factor = 0.1  # Adjust as needed (larger for more elements)
  figsize = (
      12 * scale_factor * total_elements**0.5,
      8 * scale_factor * total_elements**0.5,
  )
  plt.figure(figsize=figsize)
  node_colors = [graph.nodes[n]["color"] for n in graph.nodes()]
  nx.draw(
      graph,
      pos,
      with_labels=True,
      arrows=True,
      arrowsize=10,  # Adjust arrow size
      node_size=700,
      node_color=node_colors,
      edgecolors="#000000",
  )
  # Add title and display the graph
  plt.title(graph.graph["name"])
  plt.show()


def _assign_levels_to_nodes(graph):
  """Assigns levels to nodes in a directed acyclic graph (DAG).

  This function identifies root nodes (those with no incoming edges)
  that are either parameter, token, or literal operations. Then, it performs
  a single-source shortest path search from each root node to determine
  the level (distance) of all reachable nodes in the graph. Levels are
  assigned based on the maximum distance found for a particular node
  across all root node searches.

  Args:
    graph: A NetworkX directed acyclic graph object.

  Returns:
    A dictionary mapping nodes in the graph to their assigned levels.
  """

  root_nodes = [
      n
      for n, d in graph.in_degree()
      if d == 0 and graph.nodes[n]["op"] in ["param", "token", "literal"]
  ]
  levels = {}
  for root in root_nodes:
    for node, level in nx.single_source_shortest_path_length(
        graph, root
    ).items():
      levels[node] = max(levels.get(node, 0), level)
  return levels


def _spread_nodes_horizontally(pos):
  """Spreads nodes horizontally within each level of a node layout.

  This function takes a dictionary mapping nodes to their initial positions
  (x, y) and modifies the x-coordinate for each node to achieve a more
  aesthetic horizontal distribution within its corresponding level.

  Args:
    pos: A dictionary mapping nodes in the graph to their initial (x, y)
      positions.

  Returns:
    A dictionary with the same nodes as input but with adjusted x-coordinates
    for horizontal spreading within each level.
  """

  level_widths = {}
  for node, (_, level) in pos.items():
    level_widths.setdefault(level, []).append(node)
  for level, nodes in level_widths.items():
    width = len(nodes) * 1.2
    for i, node in enumerate(sorted(nodes)):
      # Adjust position based on level and node index within the level
      x_offset = ((i - width / 2) * 1.5) + (
          0.2 * level
      )  # Consider level for some separation
      pos[node] = (x_offset, pos[node][1])
  return pos


def get_graph_stats(graph: nx.DiGraph) -> dict[str, Any]:
  stats = {
      "number_of_nodes": graph.number_of_nodes(),
      "number_of_edges": graph.number_of_edges(),
      "in_degree_distribution": dict(graph.in_degree()),
      "out_degree_distribution": dict(graph.out_degree()),
      "longest_path_length": nx.dag_longest_path_length(graph),
  }
  return stats


def interpret_edit_paths(
    edit_paths: ir_diff.OptimizedEditPaths,
    output_path: Optional[str] = None,
) -> None:
  """Writes sorted and grouped edit paths to a file or prints them to console.

  Args:
    edit_paths: A collection of edit paths containing edit paths for nodes and
      edges.
    output_path: The path to the output file to which the edit paths are
      written.

  Returns:
    None (function writes the data to a file).
  """
  out = []
  optimal_prefix = "Optimal" if edit_paths.optimal else ""
  out.append(f"\n{optimal_prefix}Paths cost:\t{edit_paths.cost}")
  out.extend(_interpret_edit_paths(edit_paths))
  if output_path:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
      f.write("\n".join(out))
  else:
    print("\n".join(out))


def _interpret_edit_paths(edit_paths: ir_diff.OptimizedEditPaths) -> list[str]:
  """Interprets a single edit path set to a list of strings.

  Args:
    edit_paths: Edit paths to be interpreted.

  Returns:
    A list of strings describing the edit path.
  """
  node_insertions = []
  edge_insertions = []
  node_deletions = []
  edge_deletions = []
  node_updates = []
  edge_updates = []
  for node in edit_paths.node_edit_paths:
    if node[0] is None and node[1] is not None:
      node_insertions.insert(0, f"\tInsert node {node[1]}")
    elif node[1] is None and node[0] is not None:
      node_deletions.insert(0, f"\tDelete node {node[0]}")
    elif node[0] is not None and node[1] is not None and node[0] != node[1]:
      node_updates.insert(0, f"\tChange node {node[0]} to {node[1]}")
    elif node[0] == node[1]:
      continue  # keep node, no change
    else:
      print("Warning: Unsupported node edit operation")
  for edge in edit_paths.edge_edit_paths:
    if edge[0] is None and edge[1] is not None:  # Insert edge
      edge_insertions.insert(-1, f"\tInsert edge {edge[1]}")
    elif edge[1] is None and edge[0] is not None:  # Delete edge
      edge_deletions.insert(-1, f"\tDelete edge {edge[0]}")
    elif edge[0] is not None and edge[1] is not None and edge[0] != edge[1]:
      edge_updates.insert(-1, f"\tChange edge {edge[0]} to {edge[1]}")
    elif edge[0] == edge[1]:
      continue  # keep edge, no change
    else:
      print("Warning: Unsupported edge edit operation")
  return (
      sorted(edge_deletions)
      + sorted(node_deletions)
      + sorted(node_insertions)
      + sorted(node_updates)
      + sorted(edge_updates)
      + sorted(edge_insertions)
  )


def timer(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
  """Measures execution time of a function.

  This function takes a callable object (function) and its arguments and
  measures its execution time.

  Args:
    fn: A callable object (function) to be timed.
    *args: Positional arguments to be passed to the function.
    **kwargs: Keyword arguments to be passed to the function.

  Returns:
    A tuple containing the function"s result and the execution time in seconds.
  """
  start_time = time.time()
  result = fn(*args, **kwargs)
  end_time = time.time()
  duration = end_time - start_time
  return (result, duration)


def plot_optimized_edit_paths_cost_vs_time(
    costs: list[float],
    timestamps: list[float],
    output_path: str | None = None,
) -> None:
  """Plots the cost vs time series for optimized edit paths.

  Args:
    costs: A list of floats representing the cost of optimized edit paths.
    timestamps: A list of floats representing the time taken to optimize the
      corresponding edit paths in costs.
    output_path: The path to the output file to which the plot is saved.

  Returns:
    None (function generates a plot and displays it).
  """

  plt.figure(figsize=(10, 6))
  plt.grid(color="gray", linestyle="--", linewidth=0.5)
  plt.title("Optimize edit paths benchmark")
  plt.xlabel("Time (Seconds)")
  plt.ylabel("Cost")
  plt.grid(True)
  if len(costs) == 1:
    # Single cost-timestamp pair - plot a single dot
    plt.scatter(timestamps[0], costs[0], marker="o", color="black", s=50)
  else:
    for i in range(1, len(timestamps)):
      plt.plot(
          timestamps[i - 1 : i + 1],
          costs[i - 1 : i + 1],
          marker="o",
          linestyle="-",
          color="black",
          linewidth=0.5,
          markersize=8,
      )
  if output_path is not None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format="png")
