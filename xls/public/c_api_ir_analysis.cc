// Copyright 2026 The XLS Authors
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

#include "xls/public/c_api_ir_analysis.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/bit_count_query_engine.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/proc_state_range_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/union_query_engine.h"
#include "xls/public/c_api_impl_helpers.h"

namespace xls {
namespace {

absl::Status EnsureTopIsSet(Package* package) {
  if (package->GetTop().has_value()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Package %s needs a top function/proc/block.", package->name()));
}

absl::StatusOr<FunctionBase*> GetTopFunctionBase(Package* package) {
  XLS_RETURN_IF_ERROR(EnsureTopIsSet(package));
  return package->GetTop().value();
}

absl::StatusOr<Node*> GetBitsNodeById(
    const absl::flat_hash_map<int64_t, Node*>& id_to_node, int64_t node_id) {
  auto it = id_to_node.find(node_id);
  if (it == id_to_node.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Node id not found in analysis: %d", node_id));
  }
  Node* node = it->second;
  if (!node->GetType()->IsBits()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Node id %d is not bits-typed (type: %s); interval tree required.",
        node_id, node->GetType()->ToString()));
  }
  return node;
}

absl::StatusOr<std::pair<Bits, Bits>> KnownBitsMaskAndValue(
    const QueryEngine& qe, Node* node) {
  CHECK(node->GetType()->IsBits());
  int64_t bit_count = node->BitCountOrDie();
  Bits mask(bit_count);
  Bits value(bit_count);

  if (!qe.IsTracked(node)) {
    return std::make_pair(std::move(mask), std::move(value));
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_tree =
      qe.GetTernary(node);
  if (!ternary_tree.has_value()) {
    return std::make_pair(std::move(mask), std::move(value));
  }

  const TernaryVector& tvec = ternary_tree->Get({});
  CHECK_EQ(tvec.size(), bit_count);
  for (int64_t i = 0; i < bit_count; ++i) {
    switch (tvec[i]) {
      case TernaryValue::kKnownZero:
        mask.SetRange(i, i + 1, /*value=*/true);
        break;
      case TernaryValue::kKnownOne:
        mask.SetRange(i, i + 1, /*value=*/true);
        value.SetRange(i, i + 1, /*value=*/true);
        break;
      case TernaryValue::kUnknown:
        break;
    }
  }

  return std::make_pair(std::move(mask), std::move(value));
}

}  // namespace
}  // namespace xls

// The public C API uses "opaque" struct pointers but the concrete storage
// lives in C++ objects. These definitions are intentionally in the .cc file.
struct xls_ir_analysis {
  xls::Package* package = nullptr;
  xls::FunctionBase* top = nullptr;

  xls::UnionQueryEngine query_engine;
  absl::flat_hash_map<int64_t, xls::Node*> id_to_node;
};

struct xls_interval_set {
  explicit xls_interval_set(xls::IntervalSet s) : set(std::move(s)) {}
  xls::IntervalSet set;
};

extern "C" {

bool xls_ir_analysis_create_from_package(struct xls_package* p,
                                         char** error_out,
                                         struct xls_ir_analysis** out) {
  CHECK(p != nullptr);
  CHECK(error_out != nullptr);
  CHECK(out != nullptr);

  auto* analysis = new xls_ir_analysis;
  analysis->package = reinterpret_cast<xls::Package*>(p);

  absl::StatusOr<xls::FunctionBase*> top_status =
      xls::GetTopFunctionBase(analysis->package);
  if (!top_status.ok()) {
    delete analysis;
    *out = nullptr;
    *error_out = xls::ToOwnedCString(top_status.status().ToString());
    return false;
  }
  analysis->top = top_status.value();

  analysis->query_engine = xls::UnionQueryEngine::Of(
      xls::BddQueryEngine(xls::BddQueryEngine::kDefaultPathLimit),
      xls::PartialInfoQueryEngine(), xls::ProcStateRangeQueryEngine(),
      xls::BitCountQueryEngine());
  if (absl::Status st = analysis->query_engine.Populate(analysis->top).status();
      !st.ok()) {
    delete analysis;
    *out = nullptr;
    *error_out = xls::ToOwnedCString(st.ToString());
    return false;
  }

  for (xls::Node* node : analysis->top->nodes()) {
    analysis->id_to_node[node->id()] = node;
  }

  *out = analysis;
  *error_out = nullptr;
  return true;
}

void xls_ir_analysis_free(struct xls_ir_analysis* a) { delete a; }

bool xls_ir_analysis_get_known_bits_for_node_id(
    const struct xls_ir_analysis* a, int64_t node_id, char** error_out,
    struct xls_bits** known_mask_out, struct xls_bits** known_value_out) {
  CHECK(a != nullptr);
  CHECK(error_out != nullptr);
  CHECK(known_mask_out != nullptr);
  CHECK(known_value_out != nullptr);

  absl::StatusOr<xls::Node*> node =
      xls::GetBitsNodeById(a->id_to_node, node_id);
  if (!node.ok()) {
    *known_mask_out = nullptr;
    *known_value_out = nullptr;
    *error_out = xls::ToOwnedCString(node.status().ToString());
    return false;
  }

  absl::StatusOr<std::pair<xls::Bits, xls::Bits>> mask_value =
      xls::KnownBitsMaskAndValue(a->query_engine, node.value());
  if (!mask_value.ok()) {
    *known_mask_out = nullptr;
    *known_value_out = nullptr;
    *error_out = xls::ToOwnedCString(mask_value.status().ToString());
    return false;
  }

  *known_mask_out =
      reinterpret_cast<xls_bits*>(new xls::Bits(std::move(mask_value->first)));
  *known_value_out =
      reinterpret_cast<xls_bits*>(new xls::Bits(std::move(mask_value->second)));
  *error_out = nullptr;
  return true;
}

bool xls_ir_analysis_get_intervals_for_node_id(
    const struct xls_ir_analysis* a, int64_t node_id, char** error_out,
    struct xls_interval_set** intervals_out) {
  CHECK(a != nullptr);
  CHECK(error_out != nullptr);
  CHECK(intervals_out != nullptr);

  absl::StatusOr<xls::Node*> node =
      xls::GetBitsNodeById(a->id_to_node, node_id);
  if (!node.ok()) {
    *intervals_out = nullptr;
    *error_out = xls::ToOwnedCString(node.status().ToString());
    return false;
  }

  xls::IntervalSet intervals =
      a->query_engine.GetIntervals(node.value()).Get({});
  intervals.Normalize();
  *intervals_out = new xls_interval_set(std::move(intervals));
  *error_out = nullptr;
  return true;
}

int64_t xls_interval_set_get_interval_count(const struct xls_interval_set* s) {
  CHECK(s != nullptr);
  return s->set.NumberOfIntervals();
}

bool xls_interval_set_get_interval_bounds(const struct xls_interval_set* s,
                                          int64_t i, char** error_out,
                                          struct xls_bits** lo_out,
                                          struct xls_bits** hi_out) {
  CHECK(s != nullptr);
  CHECK(error_out != nullptr);
  CHECK(lo_out != nullptr);
  CHECK(hi_out != nullptr);

  if (i < 0 || i >= s->set.NumberOfIntervals()) {
    *lo_out = nullptr;
    *hi_out = nullptr;
    *error_out = xls::ToOwnedCString(
        absl::InvalidArgumentError(
            absl::StrFormat("Interval index out of range: %d", i))
            .ToString());
    return false;
  }

  const xls::Interval& interval = s->set.Intervals()[i];
  *lo_out = reinterpret_cast<xls_bits*>(new xls::Bits(interval.LowerBound()));
  *hi_out = reinterpret_cast<xls_bits*>(new xls::Bits(interval.UpperBound()));
  *error_out = nullptr;
  return true;
}

void xls_interval_set_free(struct xls_interval_set* s) { delete s; }

}  // extern "C"
