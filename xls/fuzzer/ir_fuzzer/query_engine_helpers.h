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

#ifndef XLS_FUZZER_IR_FUZZER_QUERY_ENGINE_HELPERS_H_
#define XLS_FUZZER_IR_FUZZER_QUERY_ENGINE_HELPERS_H_

#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/query_engine.h"
namespace xls {

namespace internal {
using NodeValues =
    absl::flat_hash_map<Node*, absl::flat_hash_map<std::vector<Value>, Value>>;
absl::StatusOr<NodeValues> SampleValuesWith(
    Function* f, std::vector<std::vector<Value>> args);
class ScopedCheckTestChange final : public testing::EmptyTestEventListener {
 public:
  ScopedCheckTestChange() : unit_(testing::UnitTest::GetInstance()) {
    if (!unit_) {
      return;
    }
    unit_->listeners().Append(this);
  }
  ~ScopedCheckTestChange() final {
    if (!unit_) {
      return;
    }
    unit_->listeners().Release(this);
  }
  bool new_failed() const { return new_failed_; }
  void OnTestPartResult(
      const testing::TestPartResult& result) final {
    if (result.failed()) {
      new_failed_ = true;
    }
  }

 private:
  testing::UnitTest* unit_;
  bool new_failed_ = false;
};
}  // namespace internal

template <typename QueryEngineT, typename Checker>
  requires(std::is_invocable_r_v<bool, Checker, const QueryEngineT&, Node*,
                                 const Value&> &&
           std::is_base_of_v<QueryEngine, QueryEngineT>)
void CheckQueryEngineInstanceConsistency(const FuzzPackageWithArgs& fuzz,
                                         QueryEngineT& qe, Checker check) {
  Function* the_func = fuzz.fuzz_package.p->functions().front().get();
  XLS_ASSERT_OK(qe.Populate(the_func));
  absl::flat_hash_set<std::vector<Value>> seen_values;
  std::vector<std::vector<Value>> used_values;
  for (auto v : fuzz.arg_sets) {
    if (seen_values.emplace(v).second) {
      used_values.push_back(v);
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(auto node_values,
                           internal::SampleValuesWith(the_func, used_values));
  for (auto args : used_values) {
    for (Node* n : TopoSort(the_func)) {
      internal::ScopedCheckTestChange sctc;
      if (!check(qe, n, node_values[n][args]) || sctc.new_failed()) {
        FAIL() << n << " query-engine value '"
               << static_cast<const QueryEngine&>(qe).ToString(n)
               << "' is inconsistent with known value " << node_values[n][args]
               << " for arguments {"
               << absl::StrJoin(args, ", ", [](std::string* out, Value v) {
                    absl::StrAppend(out, v.ToHumanString());
                  });
      }
    }
  }
}

template <typename QueryEngineT, typename Checker>
  requires(std::is_invocable_r_v<bool, Checker, const QueryEngineT&, Node*,
                                 const Value&> &&
           std::is_base_of_v<QueryEngine, QueryEngineT> &&
           std::is_constructible_v<QueryEngineT>)
void CheckQueryEngineConsistency(const FuzzPackageWithArgs& fuzz,
                                 Checker check) {
  QueryEngineT qe;
  CheckQueryEngineInstanceConsistency<QueryEngineT>(
      fuzz, qe, std::forward<Checker>(check));
}
}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_QUERY_ENGINE_HELPERS_H_
