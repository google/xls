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

"""A library for calculating the diff between two NetworkX graphs representing IRs."""

import dataclasses
import time
from typing import Any, Iterator, Optional

import networkx as nx


@dataclasses.dataclass(frozen=True)
class EditPath:
  """A dataclass representing edit paths."""


def _node_substitution_cost(high_cost: int):

  def _node_substitution_cost_impl(n0, n1):
    if n0['cost_attributes'] == n1['cost_attributes']:
      return 0
    return high_cost

  return _node_substitution_cost_impl


def _edge_substitution_cost(high_cost: int):

  def _edge_substitution_cost_impl(e0, e1):
    if e0['cost_attributes'] == e1['cost_attributes']:
      return 0
    return high_cost

  return _edge_substitution_cost_impl


def _unit_cost(n):
  _ = n
  return 1


def _high_cost(graph0: nx.MultiDiGraph, graph1: nx.MultiDiGraph) -> int:
  return graph0.number_of_nodes() * (graph1.number_of_nodes() + 1)


@dataclasses.dataclass(frozen=True)
class OptimizedEditPaths:
  """A dataclass representing optimized edit paths."""

  node_edit_paths: list[Any]
  edge_edit_paths: list[Any]
  cost: int
  duration: float
  optimal: bool = False


def find_optimized_edit_paths(
    before_graph: nx.MultiDiGraph,
    after_graph: nx.MultiDiGraph,
    timeout_limit: Optional[float] = None,
) -> Iterator[OptimizedEditPaths]:
  """Finds a limited set of optimized edit paths with timeout handling.

  This function utilizes the `nx.optimize_edit_paths` function from NetworkX
  to find a limited set of optimized edit paths (minimum edit distance)
  to transform one graph (graph0) into another (graph1). It considers the
  provided cost functions for node and edge substitutions, deletions, and
  insertions.

  Additionally, it allows specifying a timeout limit for the search. The
  function iterates through the returned paths, appending them to the
  `self.optimized_edit_paths` list and recording the time taken to compute
  each path in `self.optimized_timestamps`.

  NetworkX's optimize edit paths returns a list of tuples, where each tuple is
  an optimized edit path set. Each edit path set consists of three elements:
  the first is the list of node edits, the second is a list of edge edits, and
  the third is the cost of the edit path set.

  Args:
    before_graph: The first NetworkX graph to compare.
    after_graph: The second NetworkX graph to compare.
    timeout_limit: The maximum time (in seconds) to spend searching for paths.
      If None, no timeout is applied.

  Yields:
    A list of optimized edit paths.
  """
  high_cost = _high_cost(before_graph, after_graph)
  start_time = time.time()
  for node_edit_paths, edge_edit_paths, cost in nx.optimize_edit_paths(
      before_graph,
      after_graph,
      node_subst_cost=_node_substitution_cost(high_cost),
      node_del_cost=_unit_cost,
      node_ins_cost=_unit_cost,
      edge_subst_cost=_edge_substitution_cost(high_cost),
      edge_del_cost=_unit_cost,
      edge_ins_cost=_unit_cost,
      upper_bound=None,
      strictly_decreasing=True,
      timeout=timeout_limit,
  ):
    step_time = time.time()
    yield OptimizedEditPaths(
        node_edit_paths=node_edit_paths,
        edge_edit_paths=edge_edit_paths,
        cost=cost,
        duration=step_time - start_time,
        optimal=False,
    )


def find_optimal_edit_paths(
    before_graph: nx.MultiDiGraph, after_graph: nx.MultiDiGraph
) -> OptimizedEditPaths:
  """Finds the optimal edit paths between the two graphs."""
  high_cost = _high_cost(before_graph, after_graph)
  start_time = time.time()
  edit_paths, optimal_cost = nx.optimal_edit_paths(
      before_graph,
      after_graph,
      edge_match=None,
      node_subst_cost=_node_substitution_cost(high_cost),
      node_del_cost=_unit_cost,
      node_ins_cost=_unit_cost,
      edge_subst_cost=_edge_substitution_cost(high_cost),
      edge_del_cost=_unit_cost,
      edge_ins_cost=_unit_cost,
      upper_bound=None,
  )
  end_time = time.time()
  node_edit_paths = []
  edge_edit_paths = []
  for node_edit_path, edge_edit_path in edit_paths:
    node_edit_paths.extend(node_edit_path)
    edge_edit_paths.extend(edge_edit_path)
  return OptimizedEditPaths(
      node_edit_paths=node_edit_paths,
      edge_edit_paths=edge_edit_paths,
      cost=optimal_cost,
      duration=end_time - start_time,
      optimal=True,
  )
