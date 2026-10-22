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

#include "xls/passes/loop_idiom_pass.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

bool MatchInductionVar(Node* node, Param* i) {
  if (node == i) {
    return true;
  }
  if (node->op() != Op::kAdd) {
    return false;
  }
  Node* lhs = node->operand(0);
  Node* rhs = node->operand(1);
  if (lhs == i && rhs->op() == Op::kLiteral &&
      rhs->As<Literal>()->value().IsAllZeros()) {
    return true;
  }
  if (rhs == i && lhs->op() == Op::kLiteral &&
      lhs->As<Literal>()->value().IsAllZeros()) {
    return true;
  }
  return false;
}

Param* MatchInvariantParam(Node* node, Function* body) {
  Param* param = nullptr;
  if (node->Is<Param>()) {
    param = node->As<Param>();
  } else if (node->op() == Op::kZeroExt) {
    Node* operand = node->operand(0);
    if (operand->Is<Param>()) {
      param = operand->As<Param>();
    }
  }
  if (param == nullptr || param->function_base() != body) {
    return nullptr;
  }

  for (int64_t index = 2; index < static_cast<int64_t>(body->params().size());
       ++index) {
    if (body->params()[index] == param) {
      return param;
    }
  }
  return nullptr;
}

bool BodyHasSideEffects(Function* body) {
  for (Node* node : body->nodes()) {
    if (node->Is<Param>()) {
      continue;
    }
    if (OpIsSideEffecting(node->op()) || node->Is<Invoke>()) {
      return true;
    }
  }
  return false;
}

struct ArrayOverwriteLoop {
  CountedFor* loop;
  Param* i;             // body params()[0], the induction variable
  Param* result;        // body params()[1], the loop-carry array
  ArrayUpdate* update;  // the body's return value
};

std::optional<ArrayOverwriteLoop> MatchLoopStructure(CountedFor* loop) {
  if (loop->stride() != 1) {
    return std::nullopt;
  }
  if (!loop->GetType()->IsArray()) {
    return std::nullopt;
  }
  ArrayType* result_type = loop->GetType()->AsArrayOrDie();
  if (loop->trip_count() != result_type->size()) {
    return std::nullopt;
  }

  Function* body = loop->body();
  if (body->params().size() < 3) {
    return std::nullopt;
  }
  Param* i = body->params()[0];
  Param* result = body->params()[1];
  if (result->GetType() != result_type) {
    return std::nullopt;
  }

  Node* ret = body->return_value();
  if (ret == nullptr || !ret->Is<ArrayUpdate>()) {
    return std::nullopt;
  }
  ArrayUpdate* update = ret->As<ArrayUpdate>();
  if (update->array_to_update() != result || update->indices().size() != 1 ||
      !MatchInductionVar(update->indices()[0], i)) {
    return std::nullopt;
  }

  if (result->users().size() != 1 || result->users()[0] != update) {
    return std::nullopt;
  }

  return ArrayOverwriteLoop{loop, i, result, update};
}

struct ShiftIdiom {
  CountedFor* loop;
  Node* data;   // the array being shifted
  Node* shift;  // the shift amount
};

std::optional<ShiftIdiom> MatchShiftIdiom(const ArrayOverwriteLoop& structure) {
  CountedFor* loop = structure.loop;
  Function* body = loop->body();
  ArrayType* result_type = loop->GetType()->AsArrayOrDie();
  if (!result_type->element_type()->IsBits() ||
      result_type->element_type()->AsBitsOrDie()->bit_count() != 1) {
    return std::nullopt;
  }

  Node* value = structure.update->update_value();
  if (!value->Is<Select>()) {
    return std::nullopt;
  }
  Select* sel = value->As<Select>();
  if (sel->default_value().has_value() || sel->cases().size() != 2) {
    return std::nullopt;
  }
  Node* cond = sel->selector();
  if (!cond->Is<CompareOp>() ||
      (cond->op() != Op::kULt && cond->op() != Op::kUGe)) {
    return std::nullopt;
  }
  if (!MatchInductionVar(cond->operand(0), structure.i)) {
    return std::nullopt;
  }
  Node* index_case;
  Node* fill_case;
  if (cond->op() == Op::kULt) {
    index_case = sel->get_case(0);
    fill_case = sel->get_case(1);
  } else {
    index_case = sel->get_case(1);
    fill_case = sel->get_case(0);
  }
  // Only zero-filled (logical) shifts are handled.
  if (!fill_case->Is<Literal>() ||
      !fill_case->As<Literal>()->value().IsAllZeros() ||
      fill_case->GetType() != result_type->element_type()) {
    return std::nullopt;
  }

  Param* shift_param = MatchInvariantParam(cond->operand(1), body);
  if (shift_param == nullptr) {
    return std::nullopt;
  }

  if (!index_case->Is<ArrayIndex>()) {
    return std::nullopt;
  }
  ArrayIndex* array_index = index_case->As<ArrayIndex>();
  if (array_index->array()->GetType() != result_type ||
      array_index->indices().size() != 1) {
    return std::nullopt;
  }
  Node* index = array_index->indices()[0];
  if (index->op() != Op::kSub ||
      !MatchInductionVar(index->operand(0), structure.i)) {
    return std::nullopt;
  }

  if (MatchInvariantParam(index->operand(1), body) != shift_param) {
    return std::nullopt;
  }
  Param* data_param = MatchInvariantParam(array_index->array(), body);
  if (data_param == nullptr || data_param == shift_param) {
    return std::nullopt;
  }

  int64_t data_param_index = -1;
  int64_t shift_param_index = -1;
  for (int64_t index = 0; index < static_cast<int64_t>(body->params().size());
       ++index) {
    if (body->params()[index] == data_param) {
      data_param_index = index;
    }
    if (body->params()[index] == shift_param) {
      shift_param_index = index;
    }
  }
  if (data_param_index < 2 || shift_param_index < 2) {
    return std::nullopt;
  }
  Node* data_arg = loop->invariant_args()[data_param_index - 2];
  Node* shift_arg = loop->invariant_args()[shift_param_index - 2];
  return ShiftIdiom{loop, data_arg, shift_arg};
}

absl::Status RewriteShiftIdiom(const ShiftIdiom& idiom) {
  FunctionBase* f = idiom.loop->function_base();
  ArrayType* result_type = idiom.loop->GetType()->AsArrayOrDie();
  const int64_t size = result_type->size();
  const int64_t index_bit_count = Bits::MinBitCountUnsigned(size - 1);

  std::vector<Node*> packed_elements;
  packed_elements.reserve(size);
  for (int64_t k = size - 1; k >= 0; --k) {
    XLS_ASSIGN_OR_RETURN(
        Literal * index,
        f->MakeNode<Literal>(idiom.loop->loc(),
                             Value(UBits(k, index_bit_count))));
    std::vector<Node*> indices = {index};
    XLS_ASSIGN_OR_RETURN(
        Node * element,
        f->MakeNode<ArrayIndex>(idiom.loop->loc(), idiom.data, indices,
                                /*assumed_in_bounds=*/true));
    packed_elements.push_back(element);
  }
  XLS_ASSIGN_OR_RETURN(
      Node * packed,
      f->MakeNode<Concat>(idiom.loop->loc(), absl::MakeSpan(packed_elements)));
  XLS_ASSIGN_OR_RETURN(
      Node * shifted,
      f->MakeNode<BinOp>(idiom.loop->loc(), packed, idiom.shift, Op::kShll));

  std::vector<Node*> result_elements;
  result_elements.reserve(size);
  for (int64_t k = 0; k < size; ++k) {
    XLS_ASSIGN_OR_RETURN(
        Node * bit, f->MakeNode<BitSlice>(idiom.loop->loc(), shifted, k, 1));
    result_elements.push_back(bit);
  }
  XLS_ASSIGN_OR_RETURN(
      Node * result,
      f->MakeNode<Array>(idiom.loop->loc(), absl::MakeSpan(result_elements),
                         result_type->element_type()));

  XLS_RETURN_IF_ERROR(idiom.loop->ReplaceUsesWith(result));
  return f->RemoveNode(idiom.loop);
}

CountedFor* FindCountedFor(FunctionBase* f, Node* after = nullptr) {
  bool scanning = after == nullptr;
  for (Node* node : f->nodes()) {
    if (!scanning) {
      if (node == after) {
        scanning = true;
      }
      continue;
    }
    if (node->Is<CountedFor>() &&
        (f->HasImplicitUse(node) || !node->users().empty())) {
      return node->As<CountedFor>();
    }
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<bool> LoopIdiomPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  VLOG(1) << "LoopIdiomPass running on function: " << f->name();
  bool changed = false;
  CountedFor* candidate = FindCountedFor(f);
  while (candidate != nullptr) {
    std::optional<ArrayOverwriteLoop> structure = MatchLoopStructure(candidate);
    std::optional<ShiftIdiom> idiom;
    if (structure.has_value() && !BodyHasSideEffects(candidate->body())) {
      idiom = MatchShiftIdiom(*structure);
    }
    if (idiom.has_value()) {
      XLS_RETURN_IF_ERROR(RewriteShiftIdiom(*idiom));
      changed = true;
      candidate = FindCountedFor(f);
      continue;
    }
    candidate = FindCountedFor(f, candidate);
  }
  return changed;
}

}  // namespace xls
