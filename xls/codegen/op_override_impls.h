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

#ifndef XLS_CODEGEN_OP_OVERRIDE_IMPLS_H_
#define XLS_CODEGEN_OP_OVERRIDE_IMPLS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/node_representation.h"
#include "xls/codegen/op_override.h"
#include "xls/ir/op.h"

namespace xls::verilog {

absl::StatusOr<NodeRepresentation> EmitOpOverride(
    OpOverride op_override, Node* node, std::string_view name,
    absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_OP_OVERRIDE_IMPLS_H_
