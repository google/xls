// Copyright 2021 The XLS Authors
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

#include "xls/codegen/block_metrics.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"

namespace xls::verilog {
namespace {

int64_t GenerateFlopCount(Block* block) {
  int64_t count = 0;

  for (Register* reg : block->GetRegisters()) {
    Type* reg_type = reg->type();
    count += reg_type->GetFlatBitCount();
  }

  return count;
}

// Returns true if there is a combinational feedthrough path from an input port
// to an output port.
bool HasFeedthroughPass(Block* block) {
  // Nodes which have a combinational path from an input port.
  absl::flat_hash_set<Node*> input_path_nodes;
  for (Node* node : TopoSort(block)) {
    if (node->Is<InputPort>()) {
      input_path_nodes.insert(node);
      continue;
    }
    for (Node* operand : node->operands()) {
      if (operand->GetType()->GetFlatBitCount() > 0 &&
          input_path_nodes.contains(operand)) {
        input_path_nodes.insert(node);
        break;
      }
    }
    if (node->Is<OutputPort>() && input_path_nodes.contains(node)) {
      return true;
    }
  }
  return false;
}

// Sets the delay fields of `proto` based on analysis of `block`.
absl::Status SetDelayFields(Block* block, const DelayEstimator& delay_estimator,
                            BlockMetricsProto* proto) {
  // Maximum delay from input to each node.
  absl::flat_hash_map<Node*, int64_t> input_delay_map;
  // Maximum delay from a register read to each node.
  absl::flat_hash_map<Node*, int64_t> reg_delay_map;

  // Delay metrics to set on the proto.
  std::optional<int64_t> max_reg_to_reg_delay;
  std::optional<int64_t> max_input_to_reg_delay;
  std::optional<int64_t> max_reg_to_output_delay;
  std::optional<int64_t> max_feedthrough_path_delay;

  for (Node* node : TopoSort(block)) {
    if (node->Is<InputPort>()) {
      input_delay_map[node] = 0;
      continue;
    }

    auto optional_max = [](int64_t value, std::optional<int64_t> opt_value) {
      if (opt_value.has_value()) {
        return std::max(value, opt_value.value());
      }
      return value;
    };
    absl::StatusOr<int64_t> node_delay_or =
        delay_estimator.GetOperationDelayInPs(node);
    int64_t node_delay = node_delay_or.ok() ? node_delay_or.value() : 0;

    std::optional<int64_t> input_delay;
    std::optional<int64_t> reg_delay;
    for (Node* operand : node->operands()) {
      if (operand->GetType()->GetFlatBitCount() > 0) {
        if (input_delay_map.contains(operand)) {
          input_delay = optional_max(input_delay_map.at(operand) + node_delay,
                                     input_delay);
        }
        if (reg_delay_map.contains(operand)) {
          reg_delay =
              optional_max(reg_delay_map.at(operand) + node_delay, reg_delay);
        }
      }
    }
    if (input_delay.has_value()) {
      input_delay_map[node] = input_delay.value();
    }
    if (reg_delay.has_value()) {
      reg_delay_map[node] = reg_delay.value();
    }

    if (node->Is<OutputPort>()) {
      Node* data = node->operand(0);
      if (input_delay_map.contains(data)) {
        max_feedthrough_path_delay =
            optional_max(input_delay_map.at(data), max_feedthrough_path_delay);
      }
      if (reg_delay_map.contains(data)) {
        max_reg_to_output_delay =
            optional_max(reg_delay_map.at(data), max_reg_to_output_delay);
      }
      continue;
    }
    if (node->Is<RegisterRead>()) {
      // The delay of a RegisterRead is the clk-to-q delay.
      reg_delay_map[node] = node_delay;
      continue;
    }
    if (node->Is<RegisterWrite>()) {
      // The relevant register write operands for the purposes of to-register
      // paths is the data operand and the (optional) load enable. Reset is not
      // considered.
      std::vector<Node*> operands = {node->As<RegisterWrite>()->data()};
      if (node->As<RegisterWrite>()->load_enable().has_value()) {
        operands.push_back(node->As<RegisterWrite>()->load_enable().value());
      }
      for (Node* operand : operands) {
        if (input_delay_map.contains(operand)) {
          max_input_to_reg_delay =
              optional_max(input_delay_map.at(operand), max_input_to_reg_delay);
        }
        if (reg_delay_map.contains(operand)) {
          max_reg_to_reg_delay =
              optional_max(reg_delay_map.at(operand), max_reg_to_reg_delay);
        }
      }
      continue;
    }
  }

  if (max_reg_to_reg_delay.has_value()) {
    proto->set_max_reg_to_reg_delay_ps(max_reg_to_reg_delay.value());
  }
  if (max_input_to_reg_delay.has_value()) {
    proto->set_max_input_to_reg_delay_ps(max_input_to_reg_delay.value());
  }
  if (max_reg_to_output_delay.has_value()) {
    proto->set_max_reg_to_output_delay_ps(max_reg_to_output_delay.value());
  }
  if (max_feedthrough_path_delay.has_value()) {
    proto->set_max_feedthrough_path_delay_ps(
        max_feedthrough_path_delay.value());
  }

  return absl::OkStatus();
}

BomKindProto OpToBomKind(Op op) {
  switch (op) {
    case Op::kAdd:
    case Op::kSub: {
      return BOM_KIND_ADDER;
    }

    case Op::kUMul:
    case Op::kUMulp:
    case Op::kSMul:
    case Op::kSMulp: {
      return BOM_KIND_MULTIPLIER;
    }

    case Op::kUDiv:
    case Op::kSDiv:
    case Op::kUMod:
    case Op::kSMod: {
      return BOM_KIND_DIVIDER;
    }

    case Op::kEq:
    case Op::kNe:
    case Op::kUGe:
    case Op::kSGe:
    case Op::kUGt:
    case Op::kSGt:
    case Op::kULe:
    case Op::kSLe:
    case Op::kULt:
    case Op::kSLt: {
      return BOM_KIND_COMPARISON;
    }

    case Op::kAnd:
    case Op::kNand:
    case Op::kNor:
    case Op::kNot:
    case Op::kOr:
    case Op::kXor: {
      return BOM_KIND_BITWISE;
    }

    case Op::kAndReduce:
    case Op::kOrReduce:
    case Op::kXorReduce: {
      return BOM_KIND_BITWISE_REDUCTION;
    }

    case Op::kShll:
    case Op::kShrl:
    case Op::kShra:
    case Op::kDynamicBitSlice:
    case Op::kBitSliceUpdate:
    case Op::kArrayIndex:
    case Op::kArraySlice:
    case Op::kArrayUpdate: {
      return BOM_KIND_SLICE;
    }

    case Op::kSel: {
      return BOM_KIND_SELECT;
    }

    case Op::kOneHotSel: {
      return BOM_KIND_ONE_HOT_SELECT;
    }

    case Op::kPrioritySel: {
      return BOM_KIND_PRIORITY_SELECT;
    }

    case Op::kDecode: {
      return BOM_KIND_DECODE;
    }

    case Op::kEncode: {
      return BOM_KIND_ENCODE;
    }

    case Op::kOneHot: {
      return BOM_KIND_ONE_HOT;
    }

    case Op::kAssert:
    case Op::kCover:
    case Op::kAfterAll:
    case Op::kMinDelay:
    case Op::kArray:
    case Op::kArrayConcat:
    case Op::kBitSlice:
    case Op::kConcat:
    case Op::kIdentity:
    case Op::kLiteral:
    case Op::kNeg:
    case Op::kReverse:
    case Op::kSignExt:
    case Op::kTuple:
    case Op::kTupleIndex:
    case Op::kZeroExt:
    case Op::kGate:
    case Op::kTrace: {
      return BOM_KIND_INSIGNIFICANT;
    }

    case Op::kReceive:
    case Op::kSend:
    case Op::kCountedFor:
    case Op::kDynamicCountedFor:
    case Op::kInvoke:
    case Op::kInputPort:
    case Op::kOutputPort:
    case Op::kMap:
    case Op::kParam:
    case Op::kNext:
    case Op::kRegisterRead:
    case Op::kRegisterWrite:
    case Op::kInstantiationOutput:
    case Op::kInstantiationInput: {
      return BOM_KIND_MISC;
    }

      // We intentionally have no default case here so that the compiler can
      // warn when we add a new op.
  }

  LOG(FATAL) << "OpToBomKind: unsupported op: " << OpToString(op);
}

// Generate a BOM entry for a single node.
absl::Status GenerateBomEntry(Node* node, BomEntryProto* proto) {
  int64_t maximum_input_width = 0;
  for (Node* operand : node->operands()) {
    maximum_input_width =
        std::max(maximum_input_width, operand->GetType()->GetFlatBitCount());
  }

  proto->set_op(ToOpProto(node->op()));
  proto->set_kind(OpToBomKind(node->op()));
  proto->set_output_width(node->GetType()->GetFlatBitCount());
  proto->set_maximum_input_width(maximum_input_width);
  proto->set_number_of_arguments(node->operands().size());
  for (const SourceLocation& loc : node->loc().locations) {
    SourceLocationProto* loc_proto = proto->add_location();
    if (std::optional<std::string> file =
            node->package()->GetFilename(loc.fileno())) {
      loc_proto->set_file(file.value());
    }
    loc_proto->set_line(static_cast<int32_t>(loc.lineno()));
    loc_proto->set_col(static_cast<int32_t>(loc.colno()));
  }

  return absl::OkStatus();
}

// Generate a bill of materials.
absl::Status GenerateBom(Block* block, BlockMetricsProto* proto) {
  for (Node* node : block->nodes()) {
    BomEntryProto* bom_entry = proto->add_bill_of_materials();
    XLS_RETURN_IF_ERROR(GenerateBomEntry(node, bom_entry));
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<BlockMetricsProto> GenerateBlockMetrics(
    Block* block, const DelayEstimator* delay_estimator) {
  BlockMetricsProto proto;
  proto.set_flop_count(GenerateFlopCount(block));
  proto.set_feedthrough_path_exists(HasFeedthroughPass(block));

  if (delay_estimator != nullptr) {
    proto.set_delay_model(delay_estimator->name());
    XLS_RETURN_IF_ERROR(SetDelayFields(block, *delay_estimator, &proto));
  }

  XLS_RETURN_IF_ERROR(GenerateBom(block, &proto));

  return proto;
}

}  // namespace xls::verilog
