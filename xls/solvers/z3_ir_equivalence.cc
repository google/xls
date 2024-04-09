// Copyright 2023 The XLS Authors
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

#include "xls/solvers/z3_ir_equivalence.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls::solvers::z3 {
namespace {
// If "original" is an aggregate type, e.g. tuple, array, creates a nodes in
// "function" that flatten it to a Bits typed value.
//
// Since this is only used internally to compare FlattenToBits() results to
// other FlattenToBits() results, we don't need to worry about the canonical
// order of e.g. array elements in the resulting bits type.
absl::StatusOr<Node*> FlattenToBits(Function* function, Node* original) {
  Type* type = original->GetType();
  if (type->IsBits()) {
    return original;
  }
  std::vector<Node*> values;
  if (type->IsTuple()) {
    values.reserve(type->AsTupleOrDie()->size());
    for (int64_t i = 0; i < type->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * flattened,
          FlattenToBits(function,
                        function->AddNode(std::make_unique<TupleIndex>(
                            SourceInfo(), original, i,
                            absl::StrFormat("split_tuple_%s__%i__",
                                            original->GetName(), i),
                            function))));
      values.push_back(flattened);
    }
  } else {
    values.reserve(type->AsArrayOrDie()->size());
    for (int64_t i = 0; i < type->AsArrayOrDie()->size(); ++i) {
      Node* index = function->AddNode(std::make_unique<Literal>(
          SourceInfo(), Value(UBits(i, 64)),
          absl::StrFormat("split_array_index_%s__%d__", original->GetName(), i),
          function));
      XLS_ASSIGN_OR_RETURN(
          Node * flattened,
          FlattenToBits(
              function,
              function->AddNode(std::make_unique<ArrayIndex>(
                  SourceInfo(), original, absl::MakeConstSpan({index}),
                  absl::StrFormat("split_array_%s__%i__", original->GetName(),
                                  i),
                  function))));
      values.push_back(flattened);
    }
  }
  return function->AddNode(std::make_unique<Concat>(
      SourceInfo(), values,
      absl::StrFormat("split_concat_%s", original->GetName()), function));
}
}  // namespace

absl::StatusOr<ProverResult> TryProveEquivalence(Function* a, Function* b,
                                                 absl::Duration timeout) {
  std::unique_ptr<Package> to_test = std::make_unique<Package>(
      absl::StrFormat("%s_tester", a->package()->name()));
  XLS_ASSIGN_OR_RETURN(
      Function * to_test_func,
      a->Clone(absl::StrFormat("%s_test", a->name()), to_test.get()));

  XLS_RET_CHECK(
      a->return_value()->GetType()->IsEqualTo(b->return_value()->GetType()));
  XLS_RET_CHECK_EQ(a->params().size(), b->params().size());
  for (int64_t i = 0; i < a->params().size(); ++i) {
    XLS_RET_CHECK(
        a->params()[i]->GetType()->IsEqualTo(b->params()[i]->GetType()));
  }

  // Patch b into to_test. Wire up parameters to those at the same index in the
  // to_test_function.  We do this so we can test whether the two functions are
  // semantically equivalent by making a single Z3-AST function and checking a
  // single eq node's value.
  absl::flat_hash_map<Node*, Node*> node_map;
  for (Node* n : TopoSort(b)) {
    if (n->Is<Param>()) {
      XLS_ASSIGN_OR_RETURN(int64_t index, b->GetParamIndex(n->As<Param>()));
      node_map[n] = to_test_func->param(index);
      continue;
    }
    std::vector<Node*> new_ops;
    new_ops.reserve(n->operand_count());
    for (Node* op : n->operands()) {
      new_ops.push_back(node_map[op]);
    }
    XLS_ASSIGN_OR_RETURN(node_map[n],
                         n->CloneInNewFunction(new_ops, to_test_func));
  }

  // Add check, coerce any tuples/arays into bit-arrays since z3 ir-translator
  // doesn't support eq of tuples/arrays yet.
  XLS_ASSIGN_OR_RETURN(
      Node * original_result,
      FlattenToBits(to_test_func, to_test_func->return_value()));
  XLS_ASSIGN_OR_RETURN(
      Node * transformed_result,
      FlattenToBits(to_test_func, node_map[b->return_value()]));
  Node* new_ret = to_test_func->AddNode(std::make_unique<CompareOp>(
      SourceInfo(), original_result, transformed_result, Op::kEq, "TestCheck",
      to_test_func));
  XLS_RETURN_IF_ERROR(to_test_func->set_return_value(new_ret));
  // Run prover
  return TryProve(to_test_func, new_ret, Predicate::NotEqualToZero(), timeout);
}

absl::StatusOr<ProverResult> TryProveEquivalence(
    Function* original,
    const std::function<absl::Status(Package*, Function*)>& run_pass,
    absl::Duration timeout) {
  std::unique_ptr<Package> to_transform = std::make_unique<Package>(
      absl::StrFormat("%s_copy", original->package()->name()));
  XLS_ASSIGN_OR_RETURN(
      Function * to_transform_func,
      original->Clone(absl::StrFormat("%s_copy", original->name()),
                      to_transform.get()));
  XLS_RETURN_IF_ERROR(run_pass(to_transform.get(), to_transform_func));

  return TryProveEquivalence(original, to_transform_func, timeout);
}

}  // namespace xls::solvers::z3
