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

#include "xls/contrib/eco/ged_cost_functions.h"

#include <cstdint>

#include "absl/log/log.h"

namespace {

constexpr int64_t kMaxNodeCount =
    10000;  // Avoid cost accumulation overflow in GED.
constexpr int64_t kHighCost = kMaxNodeCount * 2 + 1;

}  // namespace

int NodeSubstCost(const XLSNode& n1, const XLSNode& n2) {
  if (n1.pinned || n2.pinned) {
    if (n1.pinned && n2.pinned && n1.mcs_map_index.has_value() &&
        n2.mcs_map_index.has_value() && *n1.mcs_map_index == n2.index &&
        *n2.mcs_map_index == n1.index) {
      return 0;
    }
    VLOG(3) << "Pinned nodes must map to each other (" << n1.name << ", "
            << n2.name << ")";
    return kHighCost;
  }
  return n1.label == n2.label ? 0 : kHighCost;
}

int NodeInsCost(const XLSNode& node) {
  if (node.pinned) {
    VLOG(3) << "Cannot insert pinned node (" << node.name << ")";
    return kHighCost;
  }
  return 1;
}

int NodeDelCost(const XLSNode& node) {
  if (node.pinned) {
    VLOG(3) << "Cannot delete pinned node (" << node.name << ")";
    return kHighCost;
  }
  return 1;
}

int EdgeSubstCost(const XLSEdge& e1, const XLSEdge& e2) {
  return e1.label == e2.label ? 0 : kHighCost;
}

int EdgeInsCost(const XLSEdge& e) { return 1; }

int EdgeDelCost(const XLSEdge& e) { return 1; }

ged::GEDOptions CreateUserCosts() {
  ged::GEDOptions opts{};
  opts.nodeCosts.subst = NodeSubstCost;
  opts.nodeCosts.ins = NodeInsCost;
  opts.nodeCosts.del = NodeDelCost;
  opts.edgeCosts.subst = EdgeSubstCost;
  opts.edgeCosts.ins = EdgeInsCost;
  opts.edgeCosts.del = EdgeDelCost;
  return opts;
}
