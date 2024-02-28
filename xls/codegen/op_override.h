// Copyright 2022 The XLS Authors
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

#ifndef XLS_CODEGEN_OP_OVERRIDE_H_
#define XLS_CODEGEN_OP_OVERRIDE_H_

#include <memory>
#include <string_view>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/node_representation.h"
#include "xls/codegen/vast.h"
#include "xls/ir/node.h"

namespace xls::verilog {

class ModuleBuilder;

class OpOverride {
 public:
  virtual ~OpOverride() = default;
  virtual std::unique_ptr<OpOverride> Clone() const = 0;

  virtual absl::StatusOr<NodeRepresentation> Emit(
      Node* node, std::string_view name,
      absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb) = 0;

  template <typename OpCCT>
  bool Is() const {
    return typeid(*this) == typeid(OpCCT);
  }

  // Returns a down_cast pointer of the given template argument type. CHECK
  // fails if the object is not of the given type. For example: As<Param>().
  template <typename OpCCT>
  const OpCCT* As() const {
    CHECK(Is<OpCCT>());
    return down_cast<const OpCCT*>(this);
  }
  template <typename OpCCT>
  OpCCT* As() {
    CHECK(Is<OpCCT>());
    return down_cast<OpCCT*>(this);
  }
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_OP_OVERRIDE_H_
