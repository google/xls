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

#ifndef XLS_CODEGEN_VAST_INFER_VAST_TYPES_H_
#define XLS_CODEGEN_VAST_INFER_VAST_TYPES_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/vast.h"

namespace xls {
namespace verilog {

// Builds a map of the inferred types of all expressions in the given file.
// Verilog has some unusual rules for this, described in section 11.6-11.7 of
// the Language Reference Manual. Generally, the largest operand type in an
// expression tree turns every node in the tree that type, and the LHS type can
// promote the whole RHS subtree. However, there are various exceptions.
absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> InferVastTypes(
    VerilogFile* file);

// Builds a map of the inferred types of all expressions in the given `corpus`.
// If a `VerilogFile` may have cross-file references, it is best to use this
// function to build a unified map of all types in the corpus.
absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> InferVastTypes(
    absl::Span<VerilogFile* const> corpus);

// Builds a map of the inferred types for just `expr` and its descendants. This
// should not be used to analyze a whole file in steps.
absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> InferVastTypes(
    Expression* expr);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_VAST_INFER_VAST_TYPES_H_
