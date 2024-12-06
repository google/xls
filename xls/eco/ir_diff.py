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

import time
from typing import Any, Iterator, Self

import networkx as nx


class IrDiff:
  """Calculates the diff between two NetworkX graphs representing IRs.

  This class provides methods to find the  edit distance between two graphs. It
  utilizes the `nx.optimal_edit_paths` function from NetworkX library for graph
  edit distance computation.

  Attributes:
    graph0: The first NetworkX graph to compare.
    graph1: The second NetworkX graph to compare.
    optimal_edit_paths: A tuple containing the minimum edit path and its cost
      (potentially None if not yet computed).
    optimized_edit_paths: A list of optimized edit paths with their costs.
    path_costs: A list of costs for each optimized edit path set.
    optimized_timestamps: A list of timestamps recording the time taken to
      compute each set of optimized edit paths.
    high_cost: A pre-calculated high cost value used as a replacement for
      dissimilar node/edge substitutions.
  """

  high_cost = None

  def __init__(self, graph0, graph1):
    self.graph0 = graph0
    self.graph1 = graph1
    self.optimal_edit_paths = None
    self.optimized_edit_paths = []
    self.path_costs = []
    self.optimized_timestamps = []
    IrDiff.high_cost = self.graph1.number_of_nodes() * (
        1 + self.graph0.number_of_nodes()
    )

  @classmethod
  def _node_substitution_cost(cls, n0, n1):
    if n0['cost_attributes'] == n1['cost_attributes']:
      return 0
    return cls.high_cost

  @classmethod
  def _edge_substitution_cost(cls, e0, e1):
    if e0['cost_attributes'] == e1['cost_attributes']:
      return 0
    return cls.high_cost

  @classmethod
  def _node_delete_cost(cls, n):
    _ = n
    return 1

  @classmethod
  def _edge_delete_cost(cls, n):
    _ = n
    return 1

  @classmethod
  def _node_ins_cost(cls, n):
    _ = n
    return 1

  @classmethod
  def _edge_ins_cost(cls, n):
    _ = n
    return 1

  def find_optimal_edit_paths(self: Self) -> None:
    """Finds the optimal edit paths between the two graphs."""
    self.optimal_edit_paths = nx.optimal_edit_paths(
        self.graph0,
        self.graph1,
        edge_match=None,
        node_subst_cost=IrDiff._node_substitution_cost,
        node_del_cost=IrDiff._node_delete_cost,
        node_ins_cost=IrDiff._node_ins_cost,
        edge_subst_cost=IrDiff._edge_substitution_cost,
        edge_del_cost=IrDiff._edge_delete_cost,
        edge_ins_cost=IrDiff._edge_ins_cost,
        upper_bound=None,
    )
    self.path_costs.append(self.optimal_edit_paths[1])

  def find_optimized_edit_paths(
      self: Self, timeout_limit: float | None
  ) -> Iterator[Any]:
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
      timeout_limit: The maximum time (in seconds) to spend searching for paths.
        If None, no timeout is applied.

    Yields:
      A list of optimized edit path sets.
    """
    self.path_costs = []
    start_time = time.time()
    for path_set in nx.optimize_edit_paths(
        self.graph0,
        self.graph1,
        node_subst_cost=self._node_substitution_cost,
        node_del_cost=self._node_delete_cost,
        node_ins_cost=self._node_ins_cost,
        edge_subst_cost=self._edge_substitution_cost,
        edge_del_cost=self._edge_delete_cost,
        edge_ins_cost=self._edge_ins_cost,
        upper_bound=None,
        strictly_decreasing=True,
        timeout=timeout_limit,
    ):
      self.optimized_edit_paths.append(path_set)
      self.optimized_timestamps.append(time.time() - start_time)
      self.path_costs.append(path_set[2])  # path_set[2] is the cost
      yield path_set
