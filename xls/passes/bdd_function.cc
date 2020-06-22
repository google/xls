// Copyright 2020 Google LLC
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

#include "xls/passes/bdd_function.h"

#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/ir_interpreter.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"

namespace xls {
namespace {

// Construct a BDD-based abstract evaluator. The expressions in the BDD
// saturates at a particular number of minterms. When the minterm limit is met,
// a new BDD variable is created in its place effective forgetting any
// information about the value. This avoids exponential blowup problems when
// constructing the BDD at the cost of precision. The primitive bit element of
// the abstract evaluator is a sum type consisting of a BDD node and a sentinel
// value TooManyMinterms. The TooManyMinterms value is produced if the number of
// minterms in the computed expression exceed some limit. Any logical operation
// performed with a TooManyMinterms value produces a TooManyMinterms value.
struct TooManyMinterms {};
using SaturatingBddNodeIndex = absl::variant<BddNodeIndex, TooManyMinterms>;
using SaturatingBddNodeVector = std::vector<SaturatingBddNodeIndex>;

// The AbstractEvaluator requires equals to and not equals to operations on the
// primitive element.
bool operator==(const SaturatingBddNodeIndex& a,
                const SaturatingBddNodeIndex& b) {
  if (absl::holds_alternative<TooManyMinterms>(a) ||
      absl::holds_alternative<TooManyMinterms>(b)) {
    return false;
  }
  return absl::get<BddNodeIndex>(a) == absl::get<BddNodeIndex>(b);
}

bool operator!=(const SaturatingBddNodeIndex& a,
                const SaturatingBddNodeIndex& b) {
  return !(a == b);
}

// Converts the given saturating BDD vector to a normal vector of BDD nodes. The
// input vector must not contain any TooManyMinterms values.
BddNodeVector ToBddNodeVector(const SaturatingBddNodeVector& input) {
  BddNodeVector result(input.size());
  for (int64 i = 0; i < input.size(); ++i) {
    XLS_CHECK(absl::holds_alternative<BddNodeIndex>(input[i]));
    result[i] = absl::get<BddNodeIndex>(input[i]);
  }
  return result;
}

// The abstract evaluator based on a BDD with minterm-saturating logic.
class SaturatingBddEvaluator
    : public AbstractEvaluator<SaturatingBddNodeIndex> {
 public:
  SaturatingBddEvaluator(int64 minterm_limit, BinaryDecisionDiagram* bdd)
      : minterm_limit_(minterm_limit), bdd_(bdd) {}

  SaturatingBddNodeIndex One() const override { return bdd_->one(); }

  SaturatingBddNodeIndex Zero() const override { return bdd_->zero(); }

  SaturatingBddNodeIndex Not(
      const SaturatingBddNodeIndex& input) const override {
    if (absl::holds_alternative<TooManyMinterms>(input)) {
      return TooManyMinterms();
    }
    BddNodeIndex result = bdd_->Not(absl::get<BddNodeIndex>(input));
    if (minterm_limit_ > 0 && bdd_->minterm_count(result) > minterm_limit_) {
      return TooManyMinterms();
    }
    return result;
  }

  SaturatingBddNodeIndex And(const SaturatingBddNodeIndex& a,
                             const SaturatingBddNodeIndex& b) const override {
    if (absl::holds_alternative<TooManyMinterms>(a) ||
        absl::holds_alternative<TooManyMinterms>(b)) {
      return TooManyMinterms();
    }
    BddNodeIndex result =
        bdd_->And(absl::get<BddNodeIndex>(a), absl::get<BddNodeIndex>(b));
    if (minterm_limit_ > 0 && bdd_->minterm_count(result) > minterm_limit_) {
      return TooManyMinterms();
    }
    return result;
  }

  SaturatingBddNodeIndex Or(const SaturatingBddNodeIndex& a,
                            const SaturatingBddNodeIndex& b) const override {
    if (absl::holds_alternative<TooManyMinterms>(a) ||
        absl::holds_alternative<TooManyMinterms>(b)) {
      return TooManyMinterms();
    }
    BddNodeIndex result =
        bdd_->Or(absl::get<BddNodeIndex>(a), absl::get<BddNodeIndex>(b));
    if (minterm_limit_ > 0 && bdd_->minterm_count(result) > minterm_limit_) {
      return TooManyMinterms();
    }
    return result;
  }

 private:
  int64 minterm_limit_;
  BinaryDecisionDiagram* bdd_;
};

// Returns whether the given op should be included in BDD computations.
bool ShouldEvaluate(Node* node) {
  if (!node->GetType()->IsBits()) {
    return false;
  }
  switch (node->op()) {
    // Logical ops.
    case Op::kAnd:
    case Op::kNand:
    case Op::kNor:
    case Op::kNot:
    case Op::kOr:
    case Op::kXor:
      return true;

    // Extension ops.
    case Op::kSignExt:
    case Op::kZeroExt:
      return true;

    case Op::kLiteral:
      return true;

    // Bit moving ops.
    case Op::kBitSlice:
    case Op::kDynamicBitSlice:
    case Op::kConcat:
    case Op::kReverse:
    case Op::kIdentity:
      return true;

    // Select operations.
    case Op::kOneHot:
    case Op::kOneHotSel:
    case Op::kSel:
      return true;

    // Encode/decode operations:
    case Op::kDecode:
    case Op::kEncode:
      return true;

    // Comparison operation are only expressed if at least one of the operands
    // is a literal. This avoids the potential exponential explosion of BDD
    // nodes which can occur with pathological variable ordering.
    case Op::kUGe:
    case Op::kUGt:
    case Op::kULe:
    case Op::kULt:
    case Op::kEq:
    case Op::kNe:
      return node->operand(0)->Is<Literal>() || node->operand(1)->Is<Literal>();

    // Arithmetic ops
    case Op::kAdd:
    case Op::kSMul:
    case Op::kUMul:
    case Op::kNeg:
    case Op::kSDiv:
    case Op::kSub:
    case Op::kUDiv:
      return false;

    // Reduction ops.
    case Op::kAndReduce:
    case Op::kOrReduce:
    case Op::kXorReduce:
      return true;

    // Weirdo ops.
    case Op::kArray:
    case Op::kArrayIndex:
    case Op::kArrayUpdate:
    case Op::kCountedFor:
    case Op::kInvoke:
    case Op::kMap:
    case Op::kParam:
    case Op::kTuple:
    case Op::kTupleIndex:
      return false;

    // Unsupported comparison operations.
    case Op::kSGt:
    case Op::kSGe:
    case Op::kSLe:
    case Op::kSLt:
      return false;

    // Shift operations.
    // Shifts are very intensive to compute because they decompose into many,
    // many gates and they don't seem to provide much benefit. Turn-off for now.
    // TODO(meheff): Consider enabling shifts.
    case Op::kShll:
    case Op::kShra:
    case Op::kShrl:
      return false;
  }
}

}  // namespace

/* static */ xabsl::StatusOr<std::unique_ptr<BddFunction>> BddFunction::Run(
    Function* f, int64 minterm_limit) {
  XLS_VLOG(1) << absl::StreamFormat("BddFunction::Run(%s):", f->name());
  XLS_VLOG_LINES(5, f->DumpIr());

  auto bdd_function = absl::WrapUnique(new BddFunction(f));
  SaturatingBddEvaluator evaluator(minterm_limit, &bdd_function->bdd());

  // Create and return a vector containing newly defined BDD variables.
  auto create_new_node_vector = [&](Node* n) {
    SaturatingBddNodeVector v;
    for (int64 i = 0; i < n->BitCountOrDie(); ++i) {
      v.push_back(bdd_function->bdd().NewVariable());
    }
    bdd_function->saturated_expressions_.insert(n);
    return v;
  };

  XLS_VLOG(3) << "BDD expressions:";
  absl::flat_hash_map<Node*, SaturatingBddNodeVector> values;
  for (Node* node : TopoSort(f)) {
    if (!node->GetType()->IsBits()) {
      continue;
    }
    // If we shouldn't evaluate this node or the node includes some
    // non-bits-typed operands, then just create a vector of new BDD variables
    // for this node.
    if (!ShouldEvaluate(node) ||
        std::any_of(node->operands().begin(), node->operands().end(),
                    [](Node* o) { return !o->GetType()->IsBits(); })) {
      values[node] = create_new_node_vector(node);
    } else {
      std::vector<SaturatingBddNodeVector> operand_values;
      for (Node* operand : node->operands()) {
        operand_values.push_back(values.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(
          values[node],
          AbstractEvaluate(node, operand_values, &evaluator,
                           /*default_handler=*/create_new_node_vector));

      // Associate a new BDD variable with each bit that exceeded the minterm
      // limit.
      for (SaturatingBddNodeIndex& value : values.at(node)) {
        if (absl::holds_alternative<TooManyMinterms>(value)) {
          bdd_function->saturated_expressions_.insert(node);
          value = bdd_function->bdd().NewVariable();
        }
      }
    }
    XLS_VLOG(5) << "  " << node->GetName() << ":";
    for (int64 i = 0; i < node->BitCountOrDie(); ++i) {
      XLS_VLOG(5) << absl::StreamFormat(
          "    bit %d : %s", i,
          bdd_function->bdd().ToStringDnf(
              absl::get<BddNodeIndex>(values.at(node)[i]),
              /*minterm_limit=*/15));
    }
  }

  // Copy over the vector and BDD variables into the node map which is exposed
  // via the BddFunction interface. At this point any TooManyMinterm sentinel
  // values have been replaced with new Bdd variables.
  for (const auto& pair : values) {
    bdd_function->node_map_[pair.first] = ToBddNodeVector(pair.second);
  }
  return std::move(bdd_function);
}

xabsl::StatusOr<Value> BddFunction::Evaluate(
    absl::Span<const Value> args) const {
  // Map containing the result of each node.
  absl::flat_hash_map<const Node*, Value> values;
  // Map of the BDD variable values.
  absl::flat_hash_map<BddNodeIndex, bool> bdd_variable_values;
  XLS_RET_CHECK_EQ(args.size(), func_->params().size());
  for (Node* node : TopoSort(func_)) {
    XLS_VLOG(2) << "node: " << node;
    Value result;
    if (node->Is<Param>()) {
      XLS_ASSIGN_OR_RETURN(int64 param_index,
                           func_->GetParamIndex(node->As<Param>()));
      result = args.at(param_index);
    } else if (!node->GetType()->IsBits() ||
               saturated_expressions_.contains(node)) {
      std::vector<const Value*> operand_values;
      for (Node* operand : node->operands()) {
        operand_values.push_back(&values.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(result,
                           ir_interpreter::EvaluateNode(node, operand_values));
    } else {
      const BddNodeVector& bdd_vector = node_map_.at(node);
      absl::InlinedVector<bool, 64> bits;
      for (int64 i = 0; i < bdd_vector.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(bool bit_result,
                             bdd_.Evaluate(bdd_vector[i], bdd_variable_values));
        bits.push_back(bit_result);
      }
      result = Value(Bits(bits));
    }
    values[node] = result;

    XLS_VLOG(2) << "  result: " << result;
    // Write BDD variable values into the map used for evaluation.
    if (node_map_.contains(node)) {
      const BddNodeVector& bdd_vector = node_map_.at(node);
      for (int64 i = 0; i < bdd_vector.size(); ++i) {
        if (bdd_.IsVariableBaseNode(bdd_vector.at(i))) {
          bdd_variable_values[bdd_vector.at(i)] = result.bits().Get(i);
        }
      }
    }
  }
  return values.at(func_->return_value());
}

}  // namespace xls
