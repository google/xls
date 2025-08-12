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

#include "xls/fuzzer/ir_fuzzer/query_engine_helpers.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"

namespace xls {
namespace internal {
using NodeValues =
    absl::flat_hash_map<Node*, absl::flat_hash_map<std::vector<Value>, Value>>;
namespace {
class Observer : public EvaluationObserver {
 public:
  void NodeEvaluated(Node* n, const Value& v) final { values_[n] = v; }
  const absl::flat_hash_map<Node*, Value>& values() const& { return values_; }
  absl::flat_hash_map<Node*, Value> values() && { return std::move(values_); }

 private:
  absl::flat_hash_map<Node*, Value> values_;
};
}  // namespace

absl::StatusOr<NodeValues> SampleValuesWith(
    Function* f, std::vector<std::vector<Value>> args) {
  std::vector<Observer> obs;
  obs.resize(args.size());
  NodeValues vals;
  for (const auto& arg : args) {
    Observer o;
    XLS_RETURN_IF_ERROR(InterpretFunction(f, arg, &o).status());
    for (Node* n : f->nodes()) {
      XLS_RET_CHECK(o.values().contains(n))
          << "no value of " << n << " calculated.";
      vals[n][arg] = o.values().at(n);
    }
  }
  return vals;
}

}  // namespace internal

}  // namespace xls
