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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {

// Z3-based semantics encoder that maps XLS IR operations directly to Z3 BitVec
// and Algebraic Datatype (ADT) ASTs.
class Z3SemanticsEncoder {
 public:
  explicit Z3SemanticsEncoder(Z3_context ctx);
  ~Z3SemanticsEncoder() = default;

  // Translates an XLS IR parameter to a symbolic solver variable AST.
  absl::StatusOr<Z3_ast> TranslateParam(const xls::Param* param);

  // Translates an XLS IR literal node to a concrete solver constant AST.
  absl::StatusOr<Z3_ast> TranslateLiteral(const xls::Literal* literal);

  // Translates an XLS IR node given the translated ASTs of its operands.
  absl::StatusOr<Z3_ast> TranslateNode(const xls::Node* node,
                                       absl::Span<const Z3_ast> operand_asts);

  // Encodes a multiplexer branch condition for the selected arm index.
  absl::StatusOr<Z3_ast> EncodeMuxBranchCondition(const xls::Node* mux_node,
                                                  int64_t arm_index);

  // Encodes an equality constraint between a parameter AST and a concrete
  // value.
  absl::StatusOr<Z3_ast> EncodeParamBinding(const xls::Param* param,
                                            const xls::Value& concrete_value);

  // Translates a concrete XLS Value of given Type into a Z3 AST constant.
  absl::StatusOr<Z3_ast> TranslateValue(const xls::Type* type,
                                        const xls::Value& val);

  // Gets or creates the Z3 sort corresponding to an XLS IR Type.
  Z3_sort GetTypeSort(const xls::Type& type);

  Z3_context context() const { return ctx_; }

 private:
  Z3_ast CreateTuple(const xls::TupleType* tuple_type,
                     absl::Span<const Z3_ast> elements);
  Z3_ast GetTupleElement(const xls::TupleType* tuple_type, Z3_ast tuple,
                         int64_t index);

  Z3_context ctx_;
  absl::flat_hash_map<std::string, Z3_sort> type_sort_cache_;
  absl::flat_hash_map<std::string, Z3_func_decl> tuple_constructor_cache_;
  absl::flat_hash_map<std::pair<std::string, int64_t>, Z3_func_decl>
      tuple_projector_cache_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_Z3_SEMANTICS_ENCODER_H_
