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

#ifndef XLS_PASSES_EXPRESSION_BUILDER_H_
#define XLS_PASSES_EXPRESSION_BUILDER_H_

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/bdd_query_engine.h"

namespace xls {

class ExpressionBuilder {
 public:
  explicit ExpressionBuilder(FunctionBase* func,
                             const BddQueryEngine* bdd_engine);

 private:
  // Caching previously created IR nodes
  absl::flat_hash_map<std::pair<Op, Node*>, Node*> unary_op_cache_;
  absl::flat_hash_map<std::tuple<Op, Node*, Node*>, Node*> binary_op_cache_;
  absl::flat_hash_map<std::tuple<Node*, int64_t, int64_t>, Node*>
      bitslice_cache_;

 protected:
  // The function and its analyses for which expressions will be built.
  FunctionBase* func_;
  const BddQueryEngine* bdd_engine_;

  // The temporary package used to estimate cost of expressions not yet
  // committed to being added to the function.
  Package tmp_package_;
  Function* tmp_func_;
  std::unique_ptr<BddQueryEngine> tmp_bdd_engine_;

  const BddQueryEngine* GetBddEngine(Node* node) {
    if (node->function_base() == func_) {
      return bdd_engine_;
    } else if (node->function_base() == tmp_func_) {
      return tmp_bdd_engine_.get();
    } else {
      return nullptr;
    }
  }

 public:
  bool Equivalent(Node* one, Node* other);
  bool Implies(Node* one, Node* other);
  std::vector<Node*> UniqueAndNotImplied(std::vector<Node*>& operands,
                                         bool keep_node_that_implies_other);
  absl::StatusOr<Node*> FindOrMakeBitSlice(Node* selector, int64_t start,
                                           int64_t width);
  absl::StatusOr<Node*> FindOrMakeUnaryNode(Op op, Node* operand);
  absl::StatusOr<Node*> FindOrMakeBinaryNode(Op op, Node* lhs, Node* rhs);
  absl::StatusOr<Node*> FindOrMakeNaryNode(Op op,
                                           std::vector<Node*>&& operands);
  absl::StatusOr<Node*> AndOperands(std::vector<Node*>& operands);
  absl::StatusOr<Node*> OrOperands(std::vector<Node*>& operands);

  // Returns the param in the temp function corresponding to this node, or the
  // node itself if it is already a temp function node.
  absl::StatusOr<Node*> TmpFuncNodeOrParam(Node* node);
  Function* TmpFunc() { return tmp_func_; }
};

}  // namespace xls

#endif  // XLS_PASSES_EXPRESSION_BUILDER_H_
