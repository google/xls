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

#ifndef XLS_PASSES_QUERY_ENGINE_HELPERS_H_
#define XLS_PASSES_QUERY_ENGINE_HELPERS_H_

#include <type_traits>
#include <variant>

#include "xls/common/visitor.h"
#include "xls/passes/forwarding_query_engine.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A helper to handle a possibly owned query engine of type InnerQueryEngine.
// Holds either a pointer to InnerQueryEngine or an instance of that type.
template <typename InnerQueryEngine>
  requires(std::is_base_of_v<QueryEngine, InnerQueryEngine> &&
           std::is_move_constructible_v<InnerQueryEngine> &&
           std::is_move_assignable_v<InnerQueryEngine>)
class MaybeOwnedForwardingQueryEngine : public ForwardingQueryEngine {
 public:
  explicit MaybeOwnedForwardingQueryEngine(
      std::variant<InnerQueryEngine*, InnerQueryEngine> engine)
      : engine_(std::move(engine)) {}

  // A reference to the underlying engine.
  InnerQueryEngine& engine() {
    return std::visit(
        Visitor{
            [](InnerQueryEngine* qe) -> InnerQueryEngine& { return *qe; },
            [](InnerQueryEngine& qe) -> InnerQueryEngine& { return qe; },
        },
        engine_);
  }

  // A reference to the underlying engine.
  const InnerQueryEngine& engine() const {
    return std::visit(
        Visitor{
            [](InnerQueryEngine* qe) -> const InnerQueryEngine& { return *qe; },
            [](const InnerQueryEngine& qe) -> const InnerQueryEngine& {
              return qe;
            },
        },
        engine_);
  }

 protected:
  QueryEngine& real() final { return engine(); }
  const QueryEngine& real() const final { return engine(); }

 private:
  std::variant<InnerQueryEngine*, InnerQueryEngine> engine_;
};

}  // namespace xls

#endif  // XLS_PASSES_QUERY_ENGINE_HELPERS_H_
