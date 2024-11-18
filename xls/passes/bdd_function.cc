// Copyright 2020 The XLS Authors
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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_evaluator.h"

namespace xls {
namespace {

// Returns whether the given op should be included in BDD computations.
bool ShouldEvaluate(Node* node) {
  const int64_t kMaxWidth = 64;
  auto is_wide = [](Node* n) {
    return n->GetType()->GetFlatBitCount() > kMaxWidth;
  };

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
    case Op::kConcat:
    case Op::kReverse:
    case Op::kIdentity:
      return true;
    case Op::kDynamicBitSlice:
      return !is_wide(node);

    case Op::kOneHot:
      return !is_wide(node);

    // Select operations.
    case Op::kOneHotSel:
    case Op::kPrioritySel:
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
    case Op::kSMulp:
    case Op::kUMulp:
    case Op::kNeg:
    case Op::kSDiv:
    case Op::kSub:
    case Op::kUDiv:
    case Op::kSMod:
    case Op::kUMod:
      return false;

    // Reduction ops.
    case Op::kAndReduce:
    case Op::kOrReduce:
    case Op::kXorReduce:
      return true;

    // Weirdo ops.
    case Op::kAfterAll:
    case Op::kMinDelay:
    case Op::kArray:
    case Op::kArrayConcat:
    case Op::kArrayIndex:
    case Op::kArraySlice:
    case Op::kArrayUpdate:
    case Op::kAssert:
    case Op::kCountedFor:
    case Op::kCover:
    case Op::kDynamicCountedFor:
    case Op::kGate:
    case Op::kInputPort:
    case Op::kInvoke:
    case Op::kMap:
    case Op::kOutputPort:
    case Op::kParam:
    case Op::kStateRead:
    case Op::kNext:
    case Op::kReceive:
    case Op::kRegisterRead:
    case Op::kRegisterWrite:
    case Op::kSend:
    case Op::kTrace:
    case Op::kTuple:
    case Op::kTupleIndex:
    case Op::kInstantiationInput:
    case Op::kInstantiationOutput:
      return false;

    // Unsupported comparison operations.
    case Op::kSGt:
    case Op::kSGe:
    case Op::kSLe:
    case Op::kSLt:
      return false;

    // Shift operations and related ops.
    // Shifts are very intensive to compute because they decompose into many,
    // many gates and they don't seem to provide much benefit. Turn-off for now.
    // TODO(meheff): Consider enabling shifts.
    case Op::kShll:
    case Op::kShra:
    case Op::kShrl:
    case Op::kBitSliceUpdate:
      return false;
  }
  LOG(FATAL) << "Invalid op: " << static_cast<int64_t>(node->op());
}

// Data structure which aggregates BDD performance statistics across ops.
class BddStatistics {
 public:
  // Note the compute time of a single node with the given op.
  void AddOp(Op op, const absl::Duration& duration) {
    op_durations_[op] += duration;
    absl::Duration max_duration = std::max(duration, max_op_duration_[op]);
    max_op_duration_[op] = max_duration;
    ++op_counts_[op];
  }

  absl::Duration GetTotalDuration(Op op) const {
    return op_durations_.contains(op) ? op_durations_.at(op) : absl::Duration();
  }

  absl::Duration GetMaxDuration(Op op) const {
    return max_op_duration_.contains(op) ? max_op_duration_.at(op)
                                         : absl::Duration();
  }

  std::string ToString() const {
    std::string s = "BDD compute time per op:\n";
    std::vector<Op> ops_by_duration(kAllOps.begin(), kAllOps.end());
    std::sort(ops_by_duration.begin(), ops_by_duration.end(), [&](Op a, Op b) {
      return GetTotalDuration(a) > GetTotalDuration(b);
    });
    for (Op op : ops_by_duration) {
      absl::StrAppendFormat(&s, "  %20s (%5d): %s (max %s)\n", OpToString(op),
                            op_counts_.contains(op) ? op_counts_.at(op) : 0,
                            absl::FormatDuration(GetTotalDuration(op)),
                            absl::FormatDuration(GetMaxDuration(op)));
    }
    return s;
  }

 private:
  absl::flat_hash_map<Op, absl::Duration> op_durations_;
  absl::flat_hash_map<Op, absl::Duration> max_op_duration_;
  absl::flat_hash_map<Op, int64_t> op_counts_;
};

}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<BddFunction>> BddFunction::Run(
    FunctionBase* f, int64_t path_limit,
    std::optional<std::function<bool(const Node*)>> node_filter) {
  VLOG(1) << absl::StreamFormat("BddFunction::Run(%s), %d nodes:", f->name(),
                                f->node_count());
  XLS_VLOG_LINES(5, f->DumpIr());

  auto bdd_function = absl::WrapUnique(new BddFunction(f));
  SaturatingBddEvaluator evaluator(path_limit, &bdd_function->bdd());

  // Create and return a vector containing newly defined BDD variables.
  auto create_new_node_vector = [&](Node* n) {
    SaturatingBddNodeVector v(n->BitCountOrDie());
    for (int64_t i = 0; i < n->BitCountOrDie(); ++i) {
      v[i] = bdd_function->bdd().NewVariable();
    }
    bdd_function->saturated_expressions_.insert(n);
    return v;
  };

  VLOG(3) << "BDD expressions:";
  absl::flat_hash_map<Node*, SaturatingBddNodeVector> values;
  BddStatistics bdd_stats;
  for (Node* node : TopoSort(f)) {
    VLOG(3) << "node: " << node->ToString();
    if (!node->GetType()->IsBits()) {
      VLOG(3) << "  skipping node, type is not bits: "
              << node->GetType()->ToString();
      continue;
    }

    std::optional<Stopwatch> stop_watch;
    if (VLOG_IS_ON(2)) {
      stop_watch = Stopwatch();
    }

    // If we shouldn't evaluate this node, the node is to be modeled as
    // variables, or the node includes some non-bits-typed operands, then just
    // create a vector of new BDD variables for this node.
    if (!ShouldEvaluate(node) ||
        (node_filter.has_value() && !node_filter.value()(node)) ||
        std::any_of(node->operands().begin(), node->operands().end(),
                    [](Node* o) { return !o->GetType()->IsBits(); })) {
      VLOG(3) << "  node filtered out.";
      values[node] = create_new_node_vector(node);
    } else {
      VLOG(3) << "  computing BDD value...";
      std::vector<SaturatingBddNodeVector> operand_values;
      operand_values.reserve(node->operand_count());
      for (Node* operand : node->operands()) {
        operand_values.push_back(values.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(
          values[node],
          AbstractEvaluate(node, operand_values, &evaluator,
                           /*default_handler=*/create_new_node_vector));

      // Associate a new BDD variable with each bit that exceeded the path
      // limit.
      for (SaturatingBddNodeIndex& value : values.at(node)) {
        if (std::holds_alternative<TooManyPaths>(value)) {
          bdd_function->saturated_expressions_.insert(node);
          value = bdd_function->bdd().NewVariable();
        }
      }
    }
    if (VLOG_IS_ON(5)) {
      VLOG(5) << "  " << node->GetName() << ":";
      for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
        VLOG(5) << absl::StreamFormat(
            "    bit %d : %s", i,
            bdd_function->bdd().ToStringDnf(
                std::get<BddNodeIndex>(values.at(node)[i]),
                /*minterm_limit=*/15));
      }
    }
    if (stop_watch.has_value()) {
      bdd_stats.AddOp(node->op(), stop_watch->GetElapsedTime());
    }
  }
  XLS_VLOG_LINES(2, bdd_stats.ToString());

  // Copy over the vector and BDD variables into the node map which is exposed
  // via the BddFunction interface. At this point any TooManyPaths sentinel
  // values have been replaced with new Bdd variables.
  for (const auto& pair : values) {
    bdd_function->node_map_[pair.first] = ToBddNodeVector(pair.second);
  }
  return std::move(bdd_function);
}

absl::StatusOr<Value> BddFunction::Evaluate(
    absl::Span<const Value> args) const {
  if (!func_base_->IsFunction()) {
    return absl::InvalidArgumentError(
        "Can only evaluate functions with BddFunction , not procs.");
  }
  Function* function = func_base_->AsFunctionOrDie();

  // Map containing the result of each node.
  absl::flat_hash_map<const Node*, Value> values;
  // Map of the BDD variable values.
  absl::flat_hash_map<BddNodeIndex, bool> bdd_variable_values;
  XLS_RET_CHECK_EQ(args.size(), function->params().size());
  for (Node* node : TopoSort(function)) {
    VLOG(3) << "node: " << node;
    Value result;
    if (node->Is<Param>()) {
      XLS_ASSIGN_OR_RETURN(int64_t param_index,
                           function->GetParamIndex(node->As<Param>()));
      result = args.at(param_index);
    } else if (OpIsSideEffecting(node->op()) && node->GetType()->IsToken()) {
      // Don't evaluate side-effecting ops that return tokens but conjure a
      // placeholder token so that values.at(node) doesn't have to deal with
      // missing nodes.
      result = Value::Token();
    } else if (node->Is<Cover>()) {
      // Directly set a placeholder result for covers, as these are
      // side-effecting ops that don't affect the BDD.
      result = Value::Tuple({});
    } else if (!node->GetType()->IsBits() ||
               saturated_expressions_.contains(node)) {
      std::vector<Value> operand_values;
      for (Node* operand : node->operands()) {
        operand_values.push_back(values.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(result, InterpretNode(node, operand_values));
    } else {
      const BddNodeVector& bdd_vector = node_map_.at(node);
      absl::InlinedVector<bool, 64> bits;
      for (int64_t i = 0; i < bdd_vector.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(bool bit_result,
                             bdd_.Evaluate(bdd_vector[i], bdd_variable_values));
        bits.push_back(bit_result);
      }
      result = Value(Bits(bits));
    }
    values[node] = result;

    VLOG(3) << "  result: " << result;
    // Write BDD variable values into the map used for evaluation.
    if (node_map_.contains(node)) {
      const BddNodeVector& bdd_vector = node_map_.at(node);
      for (int64_t i = 0; i < bdd_vector.size(); ++i) {
        if (bdd_.IsVariableBaseNode(bdd_vector.at(i))) {
          bdd_variable_values[bdd_vector.at(i)] = result.bits().Get(i);
        }
      }
    }
  }
  return values.at(function->return_value());
}

bool IsCheapForBdds(const Node* node) {
  // The expense of evaluating a node using a BDD can depend strongly on the
  // width of the inputs or outputs. The nodes are roughly classified into
  // different groups based on their expense with width thresholds set for each
  // group. These values are picked empirically based on benchmark results.
  constexpr int64_t kWideThreshold = 256;
  constexpr int64_t kNarrowThreshold = 16;
  constexpr int64_t kVeryNarrowThreshold = 4;

  auto is_always_cheap = [](const Node* node) {
    return node->Is<ExtendOp>() || node->Is<NaryOp>() || node->Is<BitSlice>() ||
           node->Is<Concat>() || node->Is<Literal>();
  };

  auto is_cheap_when_not_wide = [](const Node* node) {
    return IsBinarySelect(const_cast<Node*>(node)) || node->Is<UnOp>() ||
           node->Is<BitwiseReductionOp>() || node->Is<OneHot>() ||
           node->op() == Op::kEq || node->op() == Op::kNe;
  };

  auto is_cheap_when_narrow = [](const Node* node) {
    return node->Is<CompareOp>() || node->Is<OneHot>() ||
           node->Is<OneHotSelect>() || node->Is<PrioritySelect>();
  };

  int64_t width = node->GetType()->GetFlatBitCount();
  for (Node* operand : node->operands()) {
    width = std::max(operand->GetType()->GetFlatBitCount(), width);
  }

  return is_always_cheap(node) ||
         (is_cheap_when_not_wide(node) && width <= kWideThreshold) ||
         (is_cheap_when_narrow(node) && width <= kNarrowThreshold) ||
         width <= kVeryNarrowThreshold;
}

}  // namespace xls
