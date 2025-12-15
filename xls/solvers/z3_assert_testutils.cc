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

#include "xls/solvers/z3_assert_testutils.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dev_tools/extract_segment.h"
#include "xls/ir/block_testutils.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_testutils.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace xls::solvers::z3 {
namespace internal {

bool DoIsAssertClean(FunctionBase* fb, std::optional<int64_t> activations,
                     testing::MatchResultListener* result_listener) {
  if (!activations && !fb->IsFunction()) {
    *result_listener << fb->name()
                     << " is not a function but activation count is not given";
    return false;
  }
  Package test_pkg("TestPackage");
  Function* arg;
  if (fb->IsFunction()) {
    arg = fb->AsFunctionOrDie();
  } else if (fb->IsProc()) {
    auto cpy = fb->AsProcOrDie()->Clone("func_clone", &test_pkg);
    if (!cpy.ok()) {
      *result_listener << "Failed to clone proc: " << cpy.status();
      return false;
    }
    auto proc_or =
        UnrollProcToFunction(*cpy, *activations, /*include_state=*/false);
    if (!proc_or.ok()) {
      *result_listener << "Failed to translate proc to function: "
                       << proc_or.status();
      return false;
    }
    arg = *proc_or;
  } else {
    auto cpy = fb->AsBlockOrDie()->Clone("block_clone", &test_pkg);
    if (!cpy.ok()) {
      *result_listener << "Failed to clone block: " << cpy.status();
      return false;
    }
    auto block_or =
        UnrollBlockToFunction(*cpy, *activations, /*include_state=*/false,
                              /*zero_invalid_outputs=*/false);
    if (!block_or.ok()) {
      *result_listener << "Failed to translate block to function: "
                       << block_or.status();
      return false;
    }
    arg = *block_or;
  }
  absl::StatusOr<ProverResult> result = TryProveAssertClean(arg);
  bool matches = testing::ExplainMatchResult(
      absl_testing::IsOkAndHolds(IsProvenTrue()), result, result_listener);
  if (!matches && result.ok() && fb->IsFunction()) {
    testing::Test::RecordProperty(
        "failing_example", DumpWithNodeValues(arg->AsFunctionOrDie(),
                                              std::get<ProvenFalse>(*result))
                               .value_or(arg->DumpIr()));
  }
  return matches;
}

absl::StatusOr<ProverResult> TryProveAssertClean(Function* func) {
  Package test_pkg("TestPackage");
  FunctionBuilder fb("assert_checker", &test_pkg);
  std::vector<Node*> assert_predicates;
  absl::flat_hash_set<Node*> taken_nodes;
  for (Node* n : func->nodes()) {
    if (n->Is<Assert>()) {
      auto [_, added] = taken_nodes.insert(n->As<Assert>()->condition());
      if (!added) {
        continue;
      }
      assert_predicates.push_back(n->As<Assert>()->condition());
    }
  }
  if (assert_predicates.empty()) {
    testing::Test::RecordProperty("assert_clean_warning",
                                  "No asserts found in function.");
    return ProvenTrue{};
  }
  absl::flat_hash_map<Node*, Node*> old_to_new;
  XLS_ASSIGN_OR_RETURN(BValue extracted,
                       ExtractSegmentInto(fb, func, /*source_nodes=*/{},
                                          /*sink_nodes=*/assert_predicates,
                                          /*old_to_new_map=*/&old_to_new));
  absl::flat_hash_map<Node*, Node*> new_to_old;
  new_to_old.reserve(old_to_new.size());
  for (const auto& [old_node, new_node] : old_to_new) {
    new_to_old[new_node] = old_node;
  }
  Function* z3_func;
  if (assert_predicates.size() == 1) {
    // We don't generate a tuple if its just one value.
    XLS_ASSIGN_OR_RETURN(z3_func, fb.BuildWithReturnValue(extracted));
  } else {
    std::vector<BValue> vals;
    vals.reserve(assert_predicates.size());
    for (int64_t i = 0; i < assert_predicates.size(); ++i) {
      vals.push_back(fb.TupleIndex(extracted, i));
    }
    fb.And(vals);
    XLS_ASSIGN_OR_RETURN(z3_func, fb.Build());
  }
  XLS_ASSIGN_OR_RETURN(ProverResult res,
                       TryProve(z3_func, z3_func->return_value(),
                                Predicate::NotEqualToZero(), 0));
  return std::visit(
      Visitor{
          [&](ProvenFalse f) -> absl::StatusOr<ProverResult> {
            if (f.counterexample.ok()) {
              absl::flat_hash_map<const Param*, Value> mapped_counterexample;
              for (const auto& [param, value] : *f.counterexample) {
                if (new_to_old.contains(param)) {
                  mapped_counterexample[new_to_old.at(param)->As<Param>()] =
                      value;
                } else {
                  mapped_counterexample[param] = value;
                }
              }
              f.counterexample = mapped_counterexample;
            }
            return f;
          },
          [](ProvenTrue t) -> absl::StatusOr<ProverResult> { return t; },
      },
      std::move(res));
}

}  // namespace internal

}  // namespace xls::solvers::z3
