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

#ifndef XLS_CODEGEN_VAST_FOLD_VAST_CONSTANTS_H_
#define XLS_CODEGEN_VAST_FOLD_VAST_CONSTANTS_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast/vast.h"

namespace xls {
namespace verilog {

// Returns `expr` with the constants in it folded to the extent currently
// supported. The given `type_map` specifies the inferred types of
// sub-expressions (see `InferVastTypes()`); this should be provided if maximum
// bitwise accuracy to Verilog is desired.
//
// This function only fails due to unexpected conditions; inability to fold
// `expr` entirely into one `Literal`, or do anything at all, is not considered
// failure. The folding logic currently supported is focused on what is needed
// for typical data type folding.
absl::StatusOr<Expression*> FoldVastConstants(
    Expression* expr,
    const absl::flat_hash_map<Expression*, DataType*>& type_map = {});

// Like FoldVastConstants(), but folds the expression all the way to a single
// constant or fails.
absl::StatusOr<int64_t> FoldEntireVastExpr(
    Expression* expr,
    const absl::flat_hash_map<Expression*, DataType*>& type_map = {});

// Overload that folds the expressions within a `DataType` specification, to the
// point where the returned `DataType` can produce a flat bit count. If folding
// to that extent is not possible, this function will return an error.
absl::StatusOr<DataType*> FoldVastConstants(
    DataType* data_type,
    const absl::flat_hash_map<Expression*, DataType*>& type_map = {});

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_VAST_FOLD_VAST_CONSTANTS_H_
