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

#include "xls/interpreter/block_interpreter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/channel.h"
#include "xls/ir/elaborated_block_dfs_visitor.h"
#include "xls/ir/events.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {
namespace {

// An interpreter for XLS blocks.
class BlockInterpreter final : public IrInterpreter {
 public:
  BlockInterpreter(Block* block, InterpreterEvents* events,
                   std::string_view register_prefix,
                   const absl::flat_hash_map<std::string, Value>& reg_state,
                   absl::flat_hash_map<std::string, Value>& next_reg_state,
                   std::optional<EvaluationObserver*> observer)
      : IrInterpreter(nullptr, events, observer),
        register_prefix_(register_prefix),
        reg_state_(reg_state),
        next_reg_state_(next_reg_state) {
    NodeValuesMap().reserve(block->node_count());
  }

  // Ports and InstantiationInputs/Outputs are handled by the
  // ElaboratedBlockInterpreter.

  absl::Status HandleRegisterRead(RegisterRead* reg_read) override {
    std::string reg_name =
        absl::StrCat(register_prefix_, reg_read->GetRegister()->name());
    auto reg_value_iter = reg_state_.find(reg_name);
    if (reg_value_iter == reg_state_.end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Missing value for register '%s'", reg_name));
    }
    return SetValueResult(reg_read, reg_value_iter->second);
  }

  absl::Status HandleRegisterWrite(RegisterWrite* reg_write) override {
    std::string reg_name =
        absl::StrCat(register_prefix_, reg_write->GetRegister()->name());
    bool is_activated = !reg_write->load_enable().has_value() ||
                        ResolveAsBool(reg_write->load_enable().value());
    std::optional<RegisterWrite*> previously_activated_write;
    auto it = activated_register_writes_.find(reg_name);
    if (it != activated_register_writes_.end()) {
      previously_activated_write = it->second;
    }

    std::optional<Value> reset_value;
    if (reg_write->reset().has_value()) {
      bool reset_signal = ResolveAsBool(reg_write->reset().value());
      std::optional<ResetBehavior> reset_behavior =
          reg_write->function_base()->AsBlockOrDie()->GetResetBehavior();
      if ((reset_signal && !reset_behavior->active_low) ||
          (!reset_signal && reset_behavior->active_low)) {
        // Reset is activated. Next register state is the reset value.
        reset_value = reg_write->GetRegister()->reset_value();
      }
    }

    std::optional<Value> next;
    if (reset_value.has_value()) {
      next = reset_value;
    } else if (is_activated) {
      if (previously_activated_write.has_value()) {
        return absl::InternalError(absl::StrFormat(
            "Multiple writes of register `%s` activated in the same cycle: "
            "%s "
            "and %s",
            reg_name, it->second->GetName(), reg_write->GetName()));
      }
      activated_register_writes_[reg_name] = reg_write;
      next = ResolveAsValue(reg_write->data());
    }

    if (next.has_value()) {
      next_reg_state_[reg_name] = next.value();
      VLOG(3) << absl::StreamFormat("Next register value for register %s: %s",
                                    reg_name,
                                    next_reg_state_.at(reg_name).ToString());
    } else if (!previously_activated_write.has_value()) {
      // This register write has not activated nor has any other (yet). The
      // register value is preserved. This ensures that the existing register
      // value gets copied over to the next register state. This may be
      // overwritten by a later register write operation.
      next_reg_state_[reg_name] = reg_state_.at(reg_name);
    }

    // Register writes have empty tuple types.
    return SetValueResult(reg_write, Value::Tuple({}));
  }

 private:
  friend class ElaboratedBlockInterpreter;

  // The prefix to use for register names.
  const std::string_view register_prefix_;

  // The state of the registers in this iteration.
  const absl::flat_hash_map<std::string, Value>& reg_state_;

  // The next state for the registers.
  absl::flat_hash_map<std::string, Value>& next_reg_state_;

  // A register may have multiple writes so keep a track of which RegisterWrite
  // fired (predicated on) for each register during evaluation. It is an error
  // if more than one fired.
  absl::flat_hash_map<std::string, RegisterWrite*> activated_register_writes_;
};

class FifoModel {
 public:
  // FifoModels have some state that is stored in the register state map (like
  // how a block stores register values). These are the names of the registers
  // and their initial values.
  //
  // The elements register stores the elements of the FIFO.
  // We use tuples b/c empty arrays are not allowed.
  static constexpr std::string_view kElementsRegisterName = "elements";
  static Value ElementsRegisterInitialValue() { return Value::Tuple({}); }
  // The output_valid register stores state that is used to determine if the
  // output of the FIFO is valid.
  // This is only used when register_pop_outputs is true, but we add it to the
  // register state map unconditionally.
  static constexpr std::string_view kOutputValidRegisterName = "output_valid";
  static Value OutputValidRegisterInitialValue() { return Value(UBits(0, 1)); }
  static constexpr std::vector<std::string_view> kRegisterNames() {
    return {kElementsRegisterName, kOutputValidRegisterName};
  }
  static std::vector<std::pair<std::string_view, Value>>
  RegisterNamesAndInitialValues() {
    return {{kElementsRegisterName, ElementsRegisterInitialValue()},
            {kOutputValidRegisterName, OutputValidRegisterInitialValue()}};
  }

  FifoModel(Type* type, FifoConfig config, std::string_view instance_prefix_,
            const absl::flat_hash_map<std::string, Value>& reg_state,
            absl::flat_hash_map<std::string, Value>& next_reg_state)
      : type_(type),
        config_(config),
        elements_register_name_(
            absl::StrCat(instance_prefix_, kElementsRegisterName)),
        output_valid_register_name_(
            absl::StrCat(instance_prefix_, kOutputValidRegisterName)),
        reg_state_(reg_state),
        next_reg_state_(next_reg_state) {
    CHECK_OK(config.Validate());
  }

  absl::Status HandleInput(InstantiationInput* input, const Value& value) {
    if (input->port_name() == FifoInstantiation::kResetPortName) {
      reset_ = value;
    } else if (input->port_name() == FifoInstantiation::kPushValidPortName) {
      push_valid_ = value;
    } else if (input->port_name() == FifoInstantiation::kPushDataPortName) {
      push_data_ = value;
    } else if (input->port_name() == FifoInstantiation::kPopReadyPortName) {
      pop_ready_ = value;
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unexpected port '%s'", input->port_name()));
    }

    // If this fifo does not actually carry any data, we don't require
    // the `push_data_` input to have been provided. The presence/absence
    // of `push_valid_` is all that is required.
    if (type_->GetFlatBitCount() == 0) {
      push_data_ = ZeroOfType(type_);
    }

    if (!(push_valid_.has_value() && push_data_.has_value() &&
          pop_ready_.has_value() && reset_.has_value())) {
      // Not yet ready to compute next state.
      return absl::OkStatus();
    }
    // No need to do anything else if the FIFO has depth 0, we'll just pass
    // through the inputs.
    if (config_.depth() == 0) {
      NextElements() = Value::Tuple({});
      return absl::OkStatus();
    }
    XLS_RET_CHECK(Elements().IsTuple());
    absl::Span<Value const> elements = Elements().elements();

    if (reset_->IsAllOnes()) {
      NextElements() = Value::Tuple({});
      if (config_.register_pop_outputs()) {
        NextOutputValid() = OutputValidRegisterInitialValue();
      }
      return absl::OkStatus();
    }

    XLS_ASSIGN_OR_RETURN(bool push_ready, PushReady());
    XLS_ASSIGN_OR_RETURN(bool pop_valid, PopValid());
    bool do_push = push_valid_->IsAllOnes() && push_ready;
    bool do_pop = pop_ready_->IsAllOnes() && pop_valid;
    bool empty = elements.empty();
    bool immediately_pop_pushed_value =
        config_.bypass() && !config_.register_pop_outputs() && empty && do_pop;

    Value const* start = elements.begin();
    if (do_pop && !elements.empty()) {
      VLOG(2) << "FIFO " << elements_register_name_ << " popping "
              << elements.front().ToString();
      ++start;
    }
    std::vector<Value> next_elements(start, elements.end());

    if (do_push) {
      VLOG(2) << "FIFO " << elements_register_name_ << " pushing "
              << push_data_->ToString();
    }
    if (do_push && !immediately_pop_pushed_value) {
      next_elements.push_back(*push_data_);
    }
    XLS_RET_CHECK_LE(next_elements.size(), config_.depth());
    NextElements() = Value::Tuple(next_elements);
    if (config_.register_pop_outputs()) {
      bool next_output_valid = elements.size() > 1 ||
                               (elements.size() == 1 && !do_pop) ||
                               (config_.bypass() && do_push);
      NextOutputValid() =
          Value(UBits(static_cast<int64_t>(next_output_valid), 1));
    }
    return absl::OkStatus();
  }
  absl::StatusOr<Value> HandleOutput(InstantiationOutput* output) {
    XLS_RET_CHECK(Elements().IsTuple());
    absl::Span<Value const> elements = Elements().elements();
    XLS_RET_CHECK_LE(elements.size(), config_.depth());
    if (output->port_name() == FifoInstantiation::kPopValidPortName) {
      VLOG(1) << "Fifo " << elements_register_name_ << " state is "
              << elements.size() << " elements with "
              << (elements.empty() ? "<empty>" : elements.front().ToString())
              << " at head.";
      XLS_ASSIGN_OR_RETURN(bool pop_valid, PopValid());
      return Value(UBits(static_cast<int64_t>(pop_valid), 1));
    }
    if (output->port_name() == FifoInstantiation::kPopDataPortName) {
      bool empty = elements.empty();
      if (!empty) {
        return elements.front();
      }
      if (config_.bypass() && !config_.register_pop_outputs()) {
        XLS_RET_CHECK(push_data_.has_value());
        return *push_data_;
      }
      return ZeroOfType(type_);
    }
    if (output->port_name() == FifoInstantiation::kPushReadyPortName) {
      XLS_ASSIGN_OR_RETURN(bool push_ready, PushReady());
      return Value(UBits(static_cast<int64_t>(push_ready), 1));
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Unexpected port '%s'", output->port_name()));
  }

 private:
  absl::StatusOr<bool> PushReady() const {
    const Value& elements = Elements();
    bool full = elements.size() >= config_.depth();
    if (!full) {
      return true;
    }
    if (config_.register_push_outputs()) {
      return false;
    }
    XLS_RET_CHECK(pop_ready_.has_value());
    return pop_ready_->IsAllOnes();
  }

  absl::StatusOr<bool> PopValid() const {
    if (config_.register_pop_outputs()) {
      return OutputValid().IsAllOnes() && !Elements().empty();
    }
    const Value& elements = Elements();
    bool empty = elements.empty();
    if (!empty) {
      return true;
    }
    if (!config_.bypass() && empty) {
      return false;
    }
    XLS_RET_CHECK(push_valid_.has_value());
    return push_valid_->IsAllOnes();
  }

  const Value& Elements() const {
    return reg_state_.at(elements_register_name_);
  }
  Value& NextElements() { return next_reg_state_[elements_register_name_]; }

  const Value& OutputValid() const {
    return reg_state_.at(output_valid_register_name_);
  }
  Value& NextOutputValid() {
    return next_reg_state_[output_valid_register_name_];
  }

  Type* type_;
  FifoConfig config_;
  std::string elements_register_name_;
  // Only used when register_pop_outputs is true.
  std::string output_valid_register_name_;
  const absl::flat_hash_map<std::string, Value>& reg_state_;
  absl::flat_hash_map<std::string, Value>& next_reg_state_;
  std::optional<Value> push_data_;
  std::optional<Value> push_valid_;
  std::optional<Value> pop_ready_;
  std::optional<Value> reset_;
};

class ElaboratedBlockInterpreter final : public ElaboratedBlockDfsVisitor {
 public:
  ElaboratedBlockInterpreter(
      const BlockElaboration& elaboration,
      const absl::flat_hash_map<std::string, Value>& inputs,
      const absl::flat_hash_map<std::string, Value>& reg_state,
      std::optional<EvaluationObserver*> observer)
      : inputs_(inputs), reg_state_(reg_state) {
    next_reg_state_.reserve(reg_state_.size());

    for (BlockInstance* instance : elaboration.instances()) {
      if (instance->instantiation().has_value() &&
          instance->instantiation().value()->kind() ==
              InstantiationKind::kFifo) {
        auto* fifo_instantiation =
            down_cast<FifoInstantiation*>(instance->instantiation().value());
        fifo_models_.insert(
            {instance, FifoModel(fifo_instantiation->data_type(),
                                 fifo_instantiation->fifo_config(),
                                 instance->RegisterPrefix(), reg_state_,
                                 next_reg_state_)});
      } else if (instance->block().has_value()) {
        interpreters_.insert(
            {instance,
             BlockInterpreter(*instance->block(), &interpreter_events_,
                              instance->RegisterPrefix(), reg_state_,
                              next_reg_state_, observer)});
      }
    }
    CHECK_OK(SetInstance(elaboration.top()));
  }

  absl::Status HandleInputPort(InputPort* input_port,
                               BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    if (current_instance_->parent_instance().has_value()) {
      BlockInstance* parent_instance = *current_instance_->parent_instance();
      XLS_ASSIGN_OR_RETURN(std::vector<ElaboratedNode> predecessors,
                           InterInstancePredecessors(ElaboratedNode{
                               .node = input_port, .instance = instance}));

      XLS_RET_CHECK(predecessors.size() == 1 &&
                    predecessors.front().node->Is<InstantiationInput>());
      InstantiationInput* instantiation_input =
          predecessors.front().node->As<InstantiationInput>();

      auto parent_interpreter_iter = interpreters_.find(parent_instance);
      if (parent_interpreter_iter == interpreters_.end()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Missing parent interpreter for instance '%s'",
                            parent_instance->ToString()));
      }
      const BlockInterpreter& parent_interpreter =
          parent_interpreter_iter->second;
      return current_interpreter_->SetValueResult(
          input_port,
          parent_interpreter.ResolveAsValue(instantiation_input->data()));
    }
    auto port_iter = inputs_.find(input_port->GetName());
    if (port_iter == inputs_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing input for port '%s'", input_port->GetName()));
    }
    return current_interpreter_->SetValueResult(input_port, port_iter->second);
  }
  absl::Status HandleInstantiationInput(InstantiationInput* instantiation_input,
                                        BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    if (instantiation_input->instantiation()->kind() ==
        InstantiationKind::kFifo) {
      BlockInstance* fifo_instance = instance->instantiation_to_instance().at(
          instantiation_input->instantiation());
      XLS_RETURN_IF_ERROR(
          fifo_models_.at(fifo_instance)
              .HandleInput(instantiation_input,
                           current_interpreter_->NodeValuesMap().at(
                               instantiation_input->data())));
    }
    // Instantiation inputs have empty tuple types.
    return current_interpreter_->SetValueResult(instantiation_input,
                                                Value::Tuple({}));
  }
  absl::Status HandleInstantiationOutput(
      InstantiationOutput* instantiation_output,
      BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    if (instantiation_output->instantiation()->kind() ==
        InstantiationKind::kFifo) {
      BlockInstance* fifo_instance = instance->instantiation_to_instance().at(
          instantiation_output->instantiation());
      XLS_ASSIGN_OR_RETURN(
          Value fifo_output,
          fifo_models_.at(fifo_instance).HandleOutput(instantiation_output));
      return current_interpreter_->SetValueResult(instantiation_output,
                                                  fifo_output);
    }
    BlockInstance* child_instance = instance->instantiation_to_instance().at(
        instantiation_output->instantiation());
    XLS_ASSIGN_OR_RETURN(
        std::vector<ElaboratedNode> predecessors,
        InterInstancePredecessors(ElaboratedNode{.node = instantiation_output,
                                                 .instance = instance}));
    XLS_RET_CHECK(predecessors.size() == 1 &&
                  predecessors.front().node->Is<OutputPort>());
    Node* child_output_data =
        predecessors.front().node->As<OutputPort>()->operand(0);
    auto child_interpreter_iter = interpreters_.find(child_instance);
    if (child_interpreter_iter == interpreters_.end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Missing child interpreter for instance '%s'",
                          child_instance->ToString()));
    }
    const Value& child_value =
        child_interpreter_iter->second.ResolveAsValue(child_output_data);
    return current_interpreter_->SetValueResult(instantiation_output,
                                                child_value);
  }
  absl::Status HandleOutputPort(OutputPort* output_port,
                                BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    // Output ports have empty tuple types.
    return current_interpreter_->SetValueResult(output_port, Value::Tuple({}));
  }

  // The rest of these handlers simply forward to the underlying interpreter.
  absl::Status HandleAdd(BinOp* add, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleAdd(add);
  }
  absl::Status HandleAfterAll(AfterAll* after_all,
                              BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleAfterAll(after_all);
  }
  absl::Status HandleMinDelay(MinDelay* min_delay,
                              BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleMinDelay(min_delay);
  }
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce,
                               BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleAndReduce(and_reduce);
  }
  absl::Status HandleArray(Array* array, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleArray(array);
  }
  absl::Status HandleArrayConcat(ArrayConcat* array_concat,
                                 BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleArrayConcat(array_concat);
  }
  absl::Status HandleAssert(Assert* assert_op,
                            BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleAssert(assert_op);
  }
  absl::Status HandleBitSlice(BitSlice* bit_slice,
                              BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleBitSlice(bit_slice);
  }
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update,
                                    BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleBitSliceUpdate(update);
  }
  absl::Status HandleConcat(Concat* concat, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleConcat(concat);
  }
  absl::Status HandleCountedFor(CountedFor* counted_for,
                                BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleCountedFor(counted_for);
  }
  absl::Status HandleCover(Cover* cover, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleCover(cover);
  }
  absl::Status HandleDecode(Decode* decode, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleDecode(decode);
  }
  absl::Status HandleDynamicBitSlice(DynamicBitSlice* dynamic_bit_slice,
                                     BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleDynamicBitSlice(dynamic_bit_slice);
  }
  absl::Status HandleDynamicCountedFor(DynamicCountedFor* dynamic_counted_for,
                                       BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleDynamicCountedFor(dynamic_counted_for);
  }
  absl::Status HandleEncode(Encode* encode, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleEncode(encode);
  }
  absl::Status HandleEq(CompareOp* eq, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleEq(eq);
  }
  absl::Status HandleGate(Gate* gate, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleGate(gate);
  }
  absl::Status HandleIdentity(UnOp* identity,
                              BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleIdentity(identity);
  }
  absl::Status HandleInvoke(Invoke* invoke, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleInvoke(invoke);
  }
  absl::Status HandleLiteral(Literal* literal,
                             BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleLiteral(literal);
  }
  absl::Status HandleMap(Map* map, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleMap(map);
  }
  absl::Status HandleArrayIndex(ArrayIndex* index,
                                BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleArrayIndex(index);
  }
  absl::Status HandleArraySlice(ArraySlice* update,
                                BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleArraySlice(update);
  }
  absl::Status HandleArrayUpdate(ArrayUpdate* update,
                                 BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleArrayUpdate(update);
  }
  absl::Status HandleNaryAnd(NaryOp* and_op, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNaryAnd(and_op);
  }
  absl::Status HandleNaryNand(NaryOp* nand_op,
                              BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNaryNand(nand_op);
  }
  absl::Status HandleNaryNor(NaryOp* nor_op, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNaryNor(nor_op);
  }
  absl::Status HandleNaryOr(NaryOp* or_op, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNaryOr(or_op);
  }
  absl::Status HandleNaryXor(NaryOp* xor_op, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNaryXor(xor_op);
  }
  absl::Status HandleNe(CompareOp* ne, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNe(ne);
  }
  absl::Status HandleNeg(UnOp* neg, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNeg(neg);
  }
  absl::Status HandleNot(UnOp* not_op, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNot(not_op);
  }
  absl::Status HandleOneHot(OneHot* one_hot, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleOneHot(one_hot);
  }
  absl::Status HandleOneHotSel(OneHotSelect* sel,
                               BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleOneHotSel(sel);
  }
  absl::Status HandlePrioritySel(PrioritySelect* sel,
                                 BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandlePrioritySel(sel);
  }
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce,
                              BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleOrReduce(or_reduce);
  }
  absl::Status HandleParam(Param* param, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleParam(param);
  }
  absl::Status HandleStateRead(StateRead* state_read,
                               BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleStateRead(state_read);
  }
  absl::Status HandleNext(Next* next, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleNext(next);
  }
  absl::Status HandleReceive(Receive* receive,
                             BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleReceive(receive);
  }
  absl::Status HandleRegisterRead(RegisterRead* reg_read,
                                  BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleRegisterRead(reg_read);
  }
  absl::Status HandleRegisterWrite(RegisterWrite* reg_write,
                                   BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleRegisterWrite(reg_write);
  }
  absl::Status HandleReverse(UnOp* reverse, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleReverse(reverse);
  }
  absl::Status HandleSDiv(BinOp* div, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSDiv(div);
  }
  absl::Status HandleSGe(CompareOp* ge, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSGe(ge);
  }
  absl::Status HandleSGt(CompareOp* gt, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSGt(gt);
  }
  absl::Status HandleSLe(CompareOp* le, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSLe(le);
  }
  absl::Status HandleSLt(CompareOp* lt, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSLt(lt);
  }
  absl::Status HandleSMod(BinOp* mod, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSMod(mod);
  }
  absl::Status HandleSMul(ArithOp* mul, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSMul(mul);
  }
  absl::Status HandleSMulp(PartialProductOp* mul,
                           BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSMulp(mul);
  }
  absl::Status HandleSel(Select* sel, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSel(sel);
  }
  absl::Status HandleSend(Send* send, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSend(send);
  }
  absl::Status HandleShll(BinOp* shll, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleShll(shll);
  }
  absl::Status HandleShra(BinOp* shra, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleShra(shra);
  }
  absl::Status HandleShrl(BinOp* shrl, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleShrl(shrl);
  }
  absl::Status HandleSignExtend(ExtendOp* sign_ext,
                                BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSignExtend(sign_ext);
  }
  absl::Status HandleSub(BinOp* sub, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleSub(sub);
  }
  absl::Status HandleTrace(Trace* trace_op, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleTrace(trace_op);
  }
  absl::Status HandleTuple(Tuple* tuple, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleTuple(tuple);
  }
  absl::Status HandleTupleIndex(TupleIndex* index,
                                BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleTupleIndex(index);
  }
  absl::Status HandleUDiv(BinOp* div, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleUDiv(div);
  }
  absl::Status HandleUGe(CompareOp* ge, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleUGe(ge);
  }
  absl::Status HandleUGt(CompareOp* gt, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleUGt(gt);
  }
  absl::Status HandleULe(CompareOp* le, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleULe(le);
  }
  absl::Status HandleULt(CompareOp* lt, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleULt(lt);
  }
  absl::Status HandleUMod(BinOp* mod, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleUMod(mod);
  }
  absl::Status HandleUMul(ArithOp* mul, BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleUMul(mul);
  }
  absl::Status HandleUMulp(PartialProductOp* mul,
                           BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleUMulp(mul);
  }
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce,
                               BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleXorReduce(xor_reduce);
  }
  absl::Status HandleZeroExtend(ExtendOp* zero_ext,
                                BlockInstance* instance) override {
    XLS_RETURN_IF_ERROR(SetInstance(instance));
    return current_interpreter_->HandleZeroExtend(zero_ext);
  }

  absl::Status SetInstance(BlockInstance* instance) {
    if (current_instance_ == instance) {
      XLS_RET_CHECK(interpreters_.contains(instance) &&
                    current_interpreter_ == &interpreters_.at(instance));
      return absl::OkStatus();
    }
    current_instance_ = instance;
    auto it = interpreters_.find(instance);
    XLS_RET_CHECK(it != interpreters_.end());
    current_interpreter_ = &it->second;
    return absl::OkStatus();
  }

  BlockInterpreter& GetInterpreter(BlockInstance* instance) {
    return interpreters_.at(instance);
  }

  const BlockInterpreter& GetInterpreter(BlockInstance* instance) const {
    return interpreters_.at(instance);
  }

  absl::flat_hash_map<std::string, Value>&& MoveRegState() {
    return std::move(next_reg_state_);
  }

  InterpreterEvents&& MoveInterpreterEvents() {
    return std::move(interpreter_events_);
  }

 private:
  const absl::flat_hash_map<std::string, Value>& inputs_;
  const absl::flat_hash_map<std::string, Value>& reg_state_;
  absl::flat_hash_map<std::string, Value> next_reg_state_;
  InterpreterEvents interpreter_events_;
  absl::flat_hash_map<BlockInstance*, BlockInterpreter> interpreters_;
  absl::flat_hash_map<BlockInstance*, FifoModel> fifo_models_;

  // SetInstance() compares current_instance_ to its argument, so initialize
  // first.
  BlockInstance* current_instance_ = nullptr;
  BlockInterpreter* current_interpreter_ = nullptr;
};

absl::StatusOr<BlockRunResult> BlockRun(
    const absl::flat_hash_map<std::string, Value>& inputs,
    const absl::flat_hash_map<std::string, Value>& reg_state,
    const BlockElaboration& elaboration,
    std::optional<EvaluationObserver*> observer) {
  Block* top_block = *elaboration.top()->block();
  // Verify each input corresponds to an input port. The reverse check (each
  // input port has a corresponding value in `inputs`) is checked in
  // HandleInputPort.
  absl::flat_hash_set<std::string> input_port_names;
  for (InputPort* port : top_block->GetInputPorts()) {
    input_port_names.insert(port->GetName());
  }
  for (const auto& [name, value] : inputs) {
    // Empty tuples don't have data
    if (value.GetFlatBitCount() == 0) {
      continue;
    }
    if (!input_port_names.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Block has no input port '%s'", name));
    }
  }

  // Verify each register value corresponds to a register. The reverse check
  // (each register has a corresponding value in `reg_state`) is checked in
  // HandleRegisterRead.
  absl::flat_hash_set<std::string> reg_names;
  reg_names.reserve(reg_state.size());
  for (BlockInstance* inst : elaboration.instances()) {
    if (inst->instantiation().has_value() &&
        inst->instantiation().value()->kind() == InstantiationKind::kFifo) {
      for (std::string_view register_name : FifoModel::kRegisterNames()) {
        reg_names.insert(absl::StrCat(inst->RegisterPrefix(), register_name));
      }
    }
    if (!inst->block().has_value()) {
      continue;
    }
    for (Register* reg : inst->block().value()->GetRegisters()) {
      reg_names.insert(absl::StrCat(inst->RegisterPrefix(), reg->name()));
    }
  }
  for (const auto& [name, value] : reg_state) {
    if (!reg_names.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Block has no register '%s'", name));
    }
  }

  ElaboratedBlockInterpreter interpreter(elaboration, inputs, reg_state,
                                         observer);
  XLS_RETURN_IF_ERROR(elaboration.Accept(interpreter));

  BlockRunResult result;
  XLS_RETURN_IF_ERROR(interpreter.SetInstance(elaboration.top()));
  BlockInterpreter& top_interpreter =
      interpreter.GetInterpreter(elaboration.top());
  result.outputs.reserve(top_block->GetOutputPorts().size());
  for (Node* port : top_block->GetOutputPorts()) {
    result.outputs[port->GetName()] =
        top_interpreter.ResolveAsValue(port->operand(0));
  }
  result.reg_state = std::move(interpreter.MoveRegState());
  result.interpreter_events = interpreter.MoveInterpreterEvents();

  return result;
}

// A template for a generic BlockContinuation that calls a stateless evaluate
// function with all input.
template <typename Evaluate>
  requires std::is_same_v<
      absl::StatusOr<BlockRunResult>,
      std::invoke_result_t<
          Evaluate, const absl::flat_hash_map<std::string, Value>&,
          const absl::flat_hash_map<std::string, Value>&,
          const BlockElaboration&, std::optional<EvaluationObserver*>>>
class StatelessBlockContinuation final : public BlockContinuation {
 public:
  StatelessBlockContinuation(BlockElaboration&& block,
                             BlockRunResult&& initial_result,
                             Evaluate evaluator,
                             BlockEvaluator::OutputPortSampleTime sample_time)
      : elaboration_(std::move(block)),
        last_result_(std::move(initial_result)),
        evaluator_(std::move(evaluator)),
        sample_time_(sample_time) {}

  BlockEvaluator::OutputPortSampleTime sample_time() const final {
    return sample_time_;
  }

  const absl::flat_hash_map<std::string, Value>& output_ports() final {
    return last_result_.outputs;
  }

  const absl::flat_hash_map<std::string, Value>& registers() final {
    return last_result_.reg_state;
  }

  const InterpreterEvents& events() final {
    return last_result_.interpreter_events;
  }

  absl::Status RunOneCycle(
      const absl::flat_hash_map<std::string, Value>& inputs) final {
    // Run to get the next value of all registers.
    XLS_ASSIGN_OR_RETURN(
        last_result_,
        evaluator_(inputs, last_result_.reg_state, elaboration_, observer_));

    if (sample_time_ == BlockEvaluator::OutputPortSampleTime::kAfterLastClock) {
      // Propagate that register value forwards.
      XLS_ASSIGN_OR_RETURN(
          BlockRunResult raw,
          evaluator_(inputs, last_result_.reg_state, elaboration_, observer_));
      last_result_.outputs = std::move(raw.outputs);
    }
    return absl::OkStatus();
  }

  absl::Status SetRegisters(
      const absl::flat_hash_map<std::string, Value>& regs) final {
    XLS_RET_CHECK_EQ(regs.size(), last_result_.reg_state.size());
    for (const auto& [key, v] : regs) {
      XLS_RET_CHECK(last_result_.reg_state.contains(key)) << key;
      XLS_RET_CHECK(last_result_.reg_state.at(key).SameTypeAs(v))
          << "'" << key << "' is incorrect type. Expected shape to match "
          << last_result_.reg_state.at(key) << " but value " << v
          << " does not match.";
    }
    last_result_.reg_state = regs;
    return absl::OkStatus();
  }

  absl::Status SetObserver(EvaluationObserver* obs) override {
    observer_ = obs;
    return absl::OkStatus();
  }
  void ClearObserver() override { observer_.reset(); }

 private:
  BlockElaboration elaboration_;
  BlockRunResult last_result_;
  Evaluate evaluator_;
  std::optional<EvaluationObserver*> observer_;
  BlockEvaluator::OutputPortSampleTime sample_time_;
};

template <typename Evaluate>
StatelessBlockContinuation(BlockElaboration&&, BlockRunResult&&, Evaluate,
                           BlockEvaluator::OutputPortSampleTime)
    -> StatelessBlockContinuation<Evaluate>;
}  // namespace

absl::StatusOr<std::unique_ptr<BlockContinuation>>
InterpreterBlockEvaluator::MakeNewContinuation(
    BlockElaboration&& elaboration,
    const absl::flat_hash_map<std::string, Value>& initial_registers,
    BlockEvaluator::OutputPortSampleTime sample_time) const {
  // We implement fifos using some extra registers stashed in the
  // register-state. We need to add these here.
  absl::flat_hash_map<std::string, Value> ext_regs = initial_registers;
  for (BlockInstance* inst : elaboration.instances()) {
    if (inst->instantiation().has_value() &&
        inst->instantiation().value()->kind() == InstantiationKind::kFifo) {
      // TODO: google/xls#1389 - Factor FIFO state out.
      for (const auto& [name, initial_value] :
           FifoModel::RegisterNamesAndInitialValues()) {
        ext_regs.insert(
            {absl::StrCat(inst->RegisterPrefix(), name), initial_value});
      }
    }
  }

  auto* cont = new StatelessBlockContinuation(
      std::move(elaboration), BlockRunResult{.reg_state = std::move(ext_regs)},
      BlockRun, sample_time);
  return std::unique_ptr<BlockContinuation>(cont);
}

}  // namespace xls
