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

#ifndef XLS_SOLVERS_SYMEX_Z3_SEMANTICS_ENCODER_H_
#define XLS_SOLVERS_SYMEX_Z3_SEMANTICS_ENCODER_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Translates XLS IR types and operations to Z3 BitVector and Algebraic Datatype
// (ADT) AST expressions.
//
// Handles type mapping (Bits to BitVectors, composite Tuples to ADT records),
// arithmetic and comparison operator lowering, constant value translation, and
// multiplexer branch condition generation (`sel == case_idx` or `sel >=
// num_cases`).
class Z3SemanticsEncoder {
 public:
  explicit Z3SemanticsEncoder(Z3_context ctx);
  ~Z3SemanticsEncoder() = default;

  // Translates an XLS parameter node into a free symbolic Z3 variable AST.
  absl::StatusOr<Z3_ast> TranslateParam(const Param* param);

  // Translates an XLS literal node into a concrete Z3 constant AST.
  absl::StatusOr<Z3_ast> TranslateLiteral(const Literal* literal);

  // Translates an XLS IR node given the translated ASTs of its operands.
  //
  // Supported operations:
  // - Identity, Literal, Param
  // - Add, Sub, And, Or, Xor, ZeroExt, SignExt, BitSlice
  // - Eq, Ne, UGt, UGe, ULt, ULe
  // - Tuple, TupleIndex
  absl::StatusOr<Z3_ast> TranslateNode(const Node* node,
                                       absl::Span<const Z3_ast> operand_asts);

  // Encodes the multiplexer branch predicate for `arm_index`.
  //
  // For explicit cases (0 <= arm_index < cases.size()), encodes `selector ==
  // arm_index`. For the default fallback arm (arm_index == cases.size()),
  // encodes `selector >= num_cases`.
  //
  // If `selector_ast` is provided, binds the condition to that existing AST;
  // otherwise creates a free symbolic constant for the selector node.
  absl::StatusOr<Z3_ast> EncodeMuxBranchCondition(
      const Node* mux_node, int64_t arm_index, Z3_ast selector_ast = nullptr);

  // Translates a concrete `Value` of the given `type` into a Z3 constant
  // AST (bitvector or nested tuple).
  absl::StatusOr<Z3_ast> TranslateValue(const Type* type, const Value& val);

  // Returns or constructs the cached Z3 sort corresponding to an XLS Type.
  Z3_sort GetTypeSort(const Type& type);

  Z3_context context() const { return ctx_; }

 private:
  Z3_ast CreateTuple(const TupleType* tuple_type,
                     absl::Span<const Z3_ast> elements);
  Z3_ast GetTupleElement(const TupleType* tuple_type, Z3_ast tuple,
                         int64_t index);

  Z3_context ctx_;
  absl::flat_hash_map<std::string, Z3_sort> type_sort_cache_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_Z3_SEMANTICS_ENCODER_H_
