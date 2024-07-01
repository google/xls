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

#include "xls/codegen/mulp_combining_pass.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xls::verilog {
namespace {

// Tries to match `node` to the sum of the two elements of a single-use mulp
// operation. Returns the mulp node if successful, nullopt otherwise. For
// example, The node `x` will match with `m` being the returned
// PartialProductOp:
//
//   m = umulp(a, b)
//   x = add(tuple-index(m, 0), tuple-index(m, 1))
//
// Also handles the case where the add operation sums LSB slices of the partial
// product, e.g.:
//
//   m = umulp(a, b)
//   lhs = bitslice(tuple-index(m, 0), 0, N)
//   rhs = bitslice(tuple-index(m, 1), 0, N)
//   x = add(lhs, rhs)
//
// TODO(meheff): 2022/8/3 Remove bitslice matching when bitslices can be hoisted
// above mulp operations.
std::optional<PartialProductOp*> MatchMulpAdd(Node* node) {
  if (node->op() != Op::kAdd) {
    return std::nullopt;
  }
  Node* lhs = node->operand(0);
  Node* rhs = node->operand(1);
  if (lhs->Is<BitSlice>() && rhs->Is<BitSlice>()) {
    if (lhs->As<BitSlice>()->start() != 0 || !HasSingleUse(lhs) ||
        rhs->As<BitSlice>()->start() != 0 || !HasSingleUse(rhs)) {
      return std::nullopt;
    }
    lhs = lhs->operand(0);
    rhs = rhs->operand(0);
  }
  if (!lhs->Is<TupleIndex>() || !rhs->Is<TupleIndex>()) {
    return std::nullopt;
  }
  int64_t lhs_index = lhs->As<TupleIndex>()->index();
  Node* lhs_src = lhs->operand(0);
  int64_t rhs_index = rhs->As<TupleIndex>()->index();
  Node* rhs_src = rhs->operand(0);
  if (!(lhs_src == rhs_src && lhs_src->Is<PartialProductOp>() &&
        ((lhs_index == 0 && rhs_index == 1) ||
         (lhs_index == 1 && rhs_index == 0)))) {
    return std::nullopt;
  }
  PartialProductOp* mulp = lhs_src->As<PartialProductOp>();

  // Verify the only use of the mulp operation is the add (ignoring the
  // tuple-index operation which select the mulp output elements). The mulp
  // itself should have two users: tuple-index for element 0, and tuple-index
  // for element 1.
  if (mulp->users().size() != 2 || !HasSingleUse(lhs) || !HasSingleUse(rhs)) {
    return std::nullopt;
  }
  return mulp;
}

}  // namespace

absl::StatusOr<bool> MulpCombiningPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : unit->package->blocks()) {
    for (Node* node : block->nodes()) {
      if (std::optional<PartialProductOp*> mulp = MatchMulpAdd(node)) {
        XLS_RETURN_IF_ERROR(
            node->ReplaceUsesWithNew<ArithOp>(
                    mulp.value()->operand(0), mulp.value()->operand(1),
                    node->BitCountOrDie(),
                    mulp.value()->op() == Op::kSMulp ? Op::kSMul : Op::kUMul)
                .status());
        changed = true;
      }
    }
  }
  if (changed) {
    unit->GcMetadata();
  }
  return changed;
}

}  // namespace xls::verilog
