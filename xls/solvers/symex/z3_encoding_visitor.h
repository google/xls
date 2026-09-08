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

#ifndef XLS_SOLVERS_SYMEX_Z3_ENCODING_VISITOR_H_
#define XLS_SOLVERS_SYMEX_Z3_ENCODING_VISITOR_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_op_translator.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Translates an XLS IR function into Z3 BitVector and Algebraic Datatype
// (ADT) AST expressions in a single post-order pass for symbolic execution.
//
// In contrast to monolithic translators that fold multiplexers into nested
// `ite(...)` trees, `Z3EncodingVisitor` inherits from
// `xls::solvers::z3::IrTranslator` and overrides `HandleSel` and
// `HandlePrioritySel` to instantiate multiplexers (`Select`, `PrioritySelect`)
// as free symbolic SSA variables without branch constraints. It provides
// explicit branch condition (`C_k`) and arm equality (`V_mux == A_k`)
// encodings for incremental path exploration.
class Z3EncodingVisitor : public xls::solvers::z3::IrTranslator {
 public:
  explicit Z3EncodingVisitor(Z3_context ctx, Function* fn = nullptr);
  ~Z3EncodingVisitor() override = default;

  // Returns the precomputed Z3 AST corresponding to `node`. Returns nullptr if
  // not found.
  Z3_ast GetNodeAst(const Node* node) const;

  // Encodes the multiplexer branch predicate for `arm_index`.
  //
  // For explicit cases (0 <= arm_index < cases.size()), encodes
  // `selector == arm_index`.
  // For the default fallback arm (arm_index == cases.size()), encodes
  // `selector >= num_cases`.
  absl::StatusOr<Z3_ast> EncodeMuxBranchCondition(const Node* mux_node,
                                                  int64_t arm_index);

  // Encodes the SSA variable equality constraint binding the multiplexer
  // variable to the chosen arm AST: `V_mux == ast(chosen_arm)`.
  absl::StatusOr<Z3_ast> EncodeMuxArmEquality(const Node* mux_node,
                                              int64_t arm_index);

  // Overridden multiplexer handlers introducing free unconstrained SSA
  // variables:
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandlePrioritySel(PrioritySelect* psel) override;

 private:
  xls::solvers::z3::Z3OpTranslator op_translator_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_Z3_ENCODING_VISITOR_H_
