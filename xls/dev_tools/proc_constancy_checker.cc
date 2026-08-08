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

#include "xls/dev_tools/proc_constancy_checker.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_testutils.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/non_synth_removal_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_translator.h"
#include "z3/src/api/z3_api.h"

namespace xls {
namespace {

// Returns true if `n` is trivially constant by construction (i.e. a Literal or
// a tree of operations whose inputs are exclusively Literals, with no path from
// dynamic inputs like Receive or StateRead).
//
// Aggregate types that are literal are often constructed from their literal
// elements rather than making a single aggregate-typed literal. We want to
// avoid spurious "non-constant" detections in these cases.
//
// Note that `non_constant_nodes` must contain every one of `n`'s operands for
// which `IsConstantByConstruction(op)` is true in order to evaluate
// `IsConstantByConstruction(n)` correctly. This can be achieved by calling
// `IsConstantByConstruction` on the nodes in topological order, inserting
// constant nodes into the set as we go.
bool IsConstantByConstruction(
    Node* n, const absl::flat_hash_set<Node*>& constant_nodes) {
  if (n->Is<Literal>()) {
    return true;
  }
  if (n->OpIn({Op::kReceive, Op::kStateRead, Op::kParam, Op::kSend, Op::kNext,
               Op::kAssert, Op::kTrace, Op::kCover})) {
    return false;
  }
  return absl::c_all_of(n->operands(), [&](Node* operand) {
    return constant_nodes.contains(operand);
  });
}

}  // namespace

absl::Status StripNonSynthNodes(Package* package, Proc* proc) {
  OptimizationContext context;
  PassResults pass_results;
  NonSynthRemovalPass pass;
  XLS_RETURN_IF_ERROR(pass.Run(package, {}, &pass_results, context).status());

  return absl::OkStatus();
}

absl::StatusOr<std::vector<Node*>> GetNodesFilteringNonSynthAndTrivialConstants(
    Proc* proc) {
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> sorted_nodes, TopoSort(proc));
  absl::flat_hash_set<Node*> constant_nodes;
  std::vector<Node*> target_nodes;
  for (Node* n : sorted_nodes) {
    if (IsConstantByConstruction(n, constant_nodes)) {
      constant_nodes.insert(n);
      continue;
    }
    if (n->GetType()->IsToken() || n->GetType()->GetFlatBitCount() == 0) {
      continue;
    }
    if (n->OpIn({Op::kParam, Op::kStateRead, Op::kReceive, Op::kSend, Op::kNext,
                 Op::kAssert, Op::kTrace, Op::kCover})) {
      continue;
    }
    target_nodes.push_back(n);
  }
  return target_nodes;
}

absl::StatusOr<std::pair<Function*, NodeActivationMap>> UnrollProcForConstancy(
    Proc* proc, int64_t activation_count) {
  XLS_ASSIGN_OR_RETURN(
      UnrolledProc unrolled,
      UnrollProc(proc, activation_count, /*include_state=*/true,
                 Value::Tuple({Value(UBits(0xdeadbeef, 32))}),
                 /*cleanup=*/false));

  NodeActivationMap node_activations;
  for (const ActivationAction& act : unrolled.activations) {
    for (const auto& [orig_node, val] : act.node_values) {
      if (val.node() != nullptr) {
        node_activations[orig_node].push_back(val.node());
      }
    }
  }
  return std::make_pair(unrolled.function, node_activations);
}

std::vector<Z3_ast> FlattenBitsOnly(Z3_context ctx,
                                    solvers::z3::IrTranslator* translator,
                                    Type* type, Z3_ast value) {
  if (type->IsBits()) {
    return translator->FlattenValue(type, value);
  }
  if (type->IsTuple()) {
    TupleType* tuple_type = type->AsTupleOrDie();
    Z3_sort tuple_sort = Z3_get_sort(ctx, value);
    std::vector<Z3_ast> all_bits;
    for (int64_t i = 0; i < tuple_type->size(); ++i) {
      Type* elem_type = tuple_type->element_type(i);
      Z3_func_decl proj_fn = Z3_get_tuple_sort_field_decl(ctx, tuple_sort, i);
      Z3_ast elem_ast = Z3_mk_app(ctx, proj_fn, 1, &value);
      std::vector<Z3_ast> elem_bits =
          FlattenBitsOnly(ctx, translator, elem_type, elem_ast);
      all_bits.insert(all_bits.end(), elem_bits.begin(), elem_bits.end());
    }
    return all_bits;
  }
  if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    Z3_sort array_sort = Z3_get_sort(ctx, value);
    Z3_sort domain_sort = Z3_get_array_sort_domain(ctx, array_sort);
    std::vector<Z3_ast> all_bits;
    for (int64_t i = 0; i < array_type->size(); ++i) {
      Z3_ast idx_ast = Z3_mk_unsigned_int64(ctx, i, domain_sort);
      Z3_ast elem_ast = Z3_mk_select(ctx, value, idx_ast);
      std::vector<Z3_ast> elem_bits = FlattenBitsOnly(
          ctx, translator, array_type->element_type(), elem_ast);
      all_bits.insert(all_bits.end(), elem_bits.begin(), elem_bits.end());
    }
    return all_bits;
  }
  return {};
}

}  // namespace xls
