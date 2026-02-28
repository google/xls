// Copyright 2025 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_ECO_GED_COST_FUNCTIONS_H_
#define XLS_ECO_GED_COST_FUNCTIONS_H_

#include <limits>

#include "absl/log/log.h"
#include "xls/eco/ged.h"
#include "xls/eco/graph.h"

inline constexpr int64_t maxNodeCount = 10000;  // Avoid std::numeric_limits overflow due to accumulation of costs in GED.

inline constexpr int64_t highCost = maxNodeCount * 2 + 1;

inline int NodeSubstCost(const XLSNode& n1, const XLSNode& n2) {
  if (n1.pinned || n2.pinned) {
    if (n1.pinned && n2.pinned && n1.mcs_map_index.has_value() &&
        n2.mcs_map_index.has_value() && *n1.mcs_map_index == n2.index &&
        *n2.mcs_map_index == n1.index) {
      return 0;
    }
    VLOG(3) << "Pinned nodes must map to each other (" << n1.name << ", "
            << n2.name << ")";
    return highCost;
  }
  return (n1.label == n2.label ? 0 : highCost);
}

inline int NodeInsCost(const XLSNode& node) {
  if (node.pinned) {
    VLOG(3) << "Cannot insert pinned node (" << node.name << ")";
    return highCost;
  }
  return 1;
}

inline int NodeDelCost(const XLSNode& node) {
  if (node.pinned) {
    VLOG(3) << "Cannot delete pinned node (" << node.name << ")";
    return highCost;
  }
  return 1;
}

inline int EdgeSubstCost(const XLSEdge& e1, const XLSEdge& e2) {
  return e1.label == e2.label ? 0 : highCost;
}

inline int EdgeInsCost(const XLSEdge& e) { return 1; }
inline int EdgeDelCost(const XLSEdge& e) { return 1; }

inline ged::GEDOptions CreateUserCosts() {
  ged::GEDOptions opts{};
  opts.nodeCosts.subst = NodeSubstCost;
  opts.nodeCosts.ins = NodeInsCost;
  opts.nodeCosts.del = NodeDelCost;
  opts.edgeCosts.subst = EdgeSubstCost;
  opts.edgeCosts.ins = EdgeInsCost;
  opts.edgeCosts.del = EdgeDelCost;
  return opts;
}

#endif  // XLS_ECO_GED_COST_FUNCTIONS_H_
