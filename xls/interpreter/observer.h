// Copyright 2024 The XLS Authors
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

#ifndef XLS_INTERPRETER_OBSERVER_H_
#define XLS_INTERPRETER_OBSERVER_H_

#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"

namespace xls {

class RuntimeObserver;
// An observer which can be called for each node evaluated.
class EvaluationObserver {
 public:
  virtual ~EvaluationObserver() = default;
  virtual void NodeEvaluated(Node* n, const Value& v) = 0;
  // Convert this to an observer capable of accepting jit values if possible.
  virtual std::optional<RuntimeObserver*> AsRawObserver() {
    return std::nullopt;
  }
};

// Test observer that just collects every node value.
class CollectingEvaluationObserver : public EvaluationObserver {
 public:
  void NodeEvaluated(Node* n, const Value& v) final {
    values_.try_emplace(n).first->second.push_back(v);
  }

  absl::flat_hash_map<Node*, std::vector<Value>>& values() { return values_; }

 private:
  absl::flat_hash_map<Node*, std::vector<Value>> values_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_OBSERVER_H_
