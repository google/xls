// Copyright 2024 The XLS Authors
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

#include "xls/codegen/maybe_materialize_fifos_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

// Construct a Block implementing a DelayLine instantiation. The block consists
// of a number of registers equal to the latency of the delay line.
absl::StatusOr<Block*> MaterializeDelayLine(
    Block* instantiating_block, DelayLineInstantiation* instantiation) {
  Package* package = instantiating_block->package();
  Block* block = package->AddBlock(
      std::make_unique<Block>(instantiation->name(), package));
  if (instantiating_block->GetClockPort().has_value()) {
    XLS_RETURN_IF_ERROR(
        block->AddClockPort(instantiating_block->GetClockPort()->name));
  }
  if (instantiating_block->GetResetPort().has_value()) {
    XLS_RETURN_IF_ERROR(
        block
            ->AddResetPort(
                instantiating_block->GetResetPort().value()->GetName(),
                instantiating_block->GetResetBehavior().value())
            .status());
  }
  XLS_ASSIGN_OR_RETURN(
      InputPort * input_port,
      block->AddInputPort(DelayLineInstantiation::kPushDataPortName,
                          instantiation->data_type()));
  Node* data = input_port;
  for (int64_t i = 0; i < instantiation->latency(); ++i) {
    std::string reg_name =
        instantiation->channel_name().has_value()
            ? absl::StrFormat("delay_line_%s_%d",
                              instantiation->channel_name().value(), i)
            : absl::StrFormat("delay_line_%d", i);
    XLS_ASSIGN_OR_RETURN(
        Register * reg,
        block->AddRegister(reg_name, instantiation->data_type(),
                           ZeroOfType(instantiation->data_type())));
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                SourceInfo(), data,
                                /*load_enable=*/std::nullopt,
                                /*reset_value=*/block->GetResetPort(), reg)
                            .status());
    XLS_ASSIGN_OR_RETURN(
        data, block->MakeNodeWithName<RegisterRead>(SourceInfo(), reg,
                                                    /*name=*/reg->name()));
  }

  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(DelayLineInstantiation::kPopDataPortName, data)
          .status());

  return block;
}

absl::StatusOr<Block*> MaterializeFifo(NameUniquer& uniquer, Package* p,
                                       FifoInstantiation* inst,
                                       const ResetBehavior& reset_behavior,
                                       std::string_view reset_name) {
  const FifoConfig& config = inst->fifo_config();
  const int64_t depth = config.depth();
  const bool bypass = config.bypass();
  const bool register_push = config.register_push_outputs();
  const bool register_pop = config.register_pop_outputs();
  Type* u1 = p->GetBitsType(1);

  const bool have_data = inst->data_type()->GetFlatBitCount() > 0;

  // Make sure there is one extra slot at least. Bad for QOR but makes impl
  // easier since full is always tail + size == head
  const uint64_t slots_cnt = depth + 1;
  Type* ptr_type = p->GetBitsType(Bits::MinBitCountUnsigned(slots_cnt));

  std::string ty_name = (have_data) ? inst->data_type()->ToString() : "no_data";

  BlockBuilder bb(uniquer.GetSanitizedUniqueName(absl::StrFormat(
                      "fifo_for_depth_%d_ty_%s_%s%s%s", depth, ty_name,
                      bypass ? "with_bypass" : "no_bypass",
                      register_pop ? "_register_pop" : "",
                      register_push ? "_register_push" : "")),
                  p);
  XLS_RETURN_IF_ERROR(bb.AddClockPort("clk"));
  BValue reset_port = bb.ResetPort(reset_name, reset_behavior);

  BValue one_lit = bb.Literal(UBits(1, ptr_type->GetFlatBitCount()));
  BValue depth_lit = bb.Literal(UBits(depth, ptr_type->GetFlatBitCount()));
  BValue long_buf_size_lit =
      bb.Literal(UBits(slots_cnt, ptr_type->GetFlatBitCount() + 1),
                 SourceInfo(), "long_buf_size_lit");

  BValue push_valid = bb.InputPort(FifoInstantiation::kPushValidPortName, u1);
  BValue pop_ready_port =
      bb.InputPort(FifoInstantiation::kPopReadyPortName, u1);

  XLS_ASSIGN_OR_RETURN(
      Register * head_reg,
      bb.block()->AddRegisterWithZeroResetValue("head", ptr_type));
  XLS_ASSIGN_OR_RETURN(
      Register * tail_reg,
      bb.block()->AddRegisterWithZeroResetValue("tail", ptr_type));
  XLS_ASSIGN_OR_RETURN(
      Register * slots_reg,
      bb.block()->AddRegisterWithZeroResetValue("slots", ptr_type));

  BValue head = bb.RegisterRead(head_reg, SourceInfo(), "head_ptr");
  BValue tail = bb.RegisterRead(tail_reg, SourceInfo(), "tail_ptr");
  BValue slots = bb.RegisterRead(slots_reg, SourceInfo(), "slots_ptr");

  // Only used if register_pop is true.
  std::optional<Register*> pop_valid_reg;
  std::optional<BValue> pop_valid_reg_read;
  std::optional<BValue> pop_valid_load_en;

  if (register_pop) {
    XLS_ASSIGN_OR_RETURN(
        pop_valid_reg,
        bb.block()->AddRegisterWithZeroResetValue("pop_valid_reg", u1));
    pop_valid_reg_read = bb.RegisterRead(*pop_valid_reg);
    depth_lit = bb.Add(depth_lit, bb.ZeroExtend(*pop_valid_reg_read,
                                                ptr_type->GetFlatBitCount()));
    pop_valid_load_en = bb.Or(pop_ready_port, bb.Not(*pop_valid_reg_read));
  }

  auto add_mod_buf_size = [&](BValue val, BValue addend,
                              std::optional<std::string_view> name) -> BValue {
    // NB Need to be sure to not run into issues where the 2 mods here (explicit
    // umod and implicit bit-width) interfere. To avoid this just do the
    // arithmetic with an extra bit and remove it, we don't really care about
    // QOR anyway.
    // TODO(allight): Rewriting this to just make the buffer a pow-2 size seems
    // like it would be simpler.
    return bb.BitSlice(
        bb.UMod(bb.Add(bb.ZeroExtend(val, ptr_type->GetFlatBitCount() + 1),
                       bb.ZeroExtend(addend, ptr_type->GetFlatBitCount() + 1)),
                long_buf_size_lit),
        0, ptr_type->GetFlatBitCount(), SourceInfo(), name.value_or(""));
  };

  // If we aren't registering pop outputs, there's nothing special on the pop
  // side and we directly forward the pop ready port. If we do register pop
  // outputs, there's a 1-element FIFO-like state machine between the pop
  // outputs and the buffer. This signal should include pop_valid, but it hasn't
  // been computed yet and we don't want a circular reference. Later, we'll
  // update buf_pop_ready to be AND(pop_valid, *pop_valid_load_en)
  BValue buf_pop_ready = pop_ready_port;
  if (register_pop) {
    buf_pop_ready = bb.Or(*pop_valid_load_en, pop_ready_port, SourceInfo(),
                          "buf_pop_ready");
  }

  // Output the current state.
  BValue is_full_bool =
      bb.Eq(slots, bb.Literal(UBits(depth, ptr_type->GetFlatBitCount())),
            SourceInfo(), "is_full_bool");
  BValue not_full_bool = bb.Not(is_full_bool);
  BValue can_do_push =
      bb.Or(not_full_bool, buf_pop_ready, SourceInfo(), "can_do_push");
  // Ready to take things if we are not full.
  BValue buf_not_empty_bool = bb.Ne(head, tail);

  // NB we don't bother clearing data after a pop.
  BValue next_tail_value_if_pop_occurs =
      add_mod_buf_size(tail, one_lit, "next_tail_if_pop");
  BValue next_head_value_if_push_occurs =
      add_mod_buf_size(head, one_lit, "next_head_if_push");

  BValue push_ready;
  BValue pop_valid;
  BValue did_pop_occur_bool;
  BValue did_push_occur_bool;

  // Only used if bypass is true.
  std::optional<BValue> is_empty_bool = std::nullopt;

  if (bypass) {
    // NB No need to handle the 'full and bypass and both read & write case
    // specially since we have an extra slot to store those values in.
    BValue bypass_possible = bb.And(buf_pop_ready, push_valid);
    BValue is_pushable = bb.Or(can_do_push, bypass_possible);
    BValue is_popable = bb.Or(buf_not_empty_bool, bypass_possible);
    pop_valid = bb.Or(buf_not_empty_bool, push_valid);
    push_ready = bb.Or(can_do_push, buf_pop_ready);
    is_empty_bool = bb.Eq(head, tail);
    BValue did_no_write_bypass_occur = bb.And(*is_empty_bool, bypass_possible);
    BValue did_visible_push_occur_bool = bb.And(is_pushable, push_valid);
    if (register_pop) {
      buf_pop_ready = bb.And(buf_pop_ready, pop_valid);
    }
    BValue did_visible_pop_occur_bool = bb.And(is_popable, buf_pop_ready);
    did_pop_occur_bool =
        bb.And(did_visible_pop_occur_bool, bb.Not(did_no_write_bypass_occur),
               SourceInfo(), "did_pop_occur");
    did_push_occur_bool =
        bb.And(did_visible_push_occur_bool, bb.Not(did_no_write_bypass_occur));
  } else {
    push_ready = can_do_push;
    pop_valid = buf_not_empty_bool;
    did_pop_occur_bool =
        bb.And(pop_valid, buf_pop_ready, SourceInfo(), "did_pop_occur");
    did_push_occur_bool = bb.And(push_ready, push_valid);
  }
  if (register_push) {
    push_ready = not_full_bool;
    did_push_occur_bool = bb.And(did_push_occur_bool, not_full_bool,
                                 SourceInfo(), "did_push_occur");
  }
  if (register_pop) {
    bb.RegisterWrite(*pop_valid_reg, pop_valid,
                     /*load_enable=*/pop_valid_load_en, reset_port);

    pop_valid = *pop_valid_reg_read;
  }
  bb.OutputPort(FifoInstantiation::kPushReadyPortName, push_ready);

  // Empty is head == tail so ne means there's something to pop.
  bb.OutputPort(FifoInstantiation::kPopValidPortName, pop_valid);

  BValue pushed = bb.And(push_ready, push_valid, SourceInfo(), "pushed");
  BValue popped = bb.And(pop_ready_port, pop_valid, SourceInfo(), "popped");
  BValue slots_next = bb.Select(
      pushed, {bb.Select(popped, {slots, bb.Subtract(slots, one_lit)}),
               bb.Select(popped, {bb.Add(slots, one_lit), slots})});

  bb.RegisterWrite(head_reg, next_head_value_if_push_occurs,
                   /*load_enable=*/did_push_occur_bool, reset_port);
  bb.RegisterWrite(tail_reg, next_tail_value_if_pop_occurs,
                   /*load_enable=*/did_pop_occur_bool, reset_port);
  bb.RegisterWrite(slots_reg, slots_next, /*load_enable=*/std::nullopt,
                   reset_port);

  // Only generate data path and I/O if the data type is of non-zero wdith.
  if (have_data) {
    Type* ty = inst->data_type();
    Type* buf_type = p->GetArrayType(slots_cnt, ty);

    BValue push_data = bb.InputPort(FifoInstantiation::kPushDataPortName, ty);

    XLS_ASSIGN_OR_RETURN(
        Register * buf_reg,
        bb.block()->AddRegisterWithZeroResetValue("buf", buf_type));
    BValue buf = bb.RegisterRead(buf_reg);

    // Only used if register_pop is true.
    std::optional<Register*> pop_data_reg;
    std::optional<BValue> pop_data_reg_read;

    if (register_pop) {
      XLS_ASSIGN_OR_RETURN(
          pop_data_reg,
          bb.block()->AddRegisterWithZeroResetValue("pop_data_reg", ty));
      pop_data_reg_read = bb.RegisterRead(*pop_data_reg);
    }

    BValue current_queue_tail = bb.ArrayIndex(buf, {tail});

    BValue next_buf_value_if_push_occurs =
        bb.ArrayUpdate(buf, push_data, {head});

    BValue pop_data_value;
    if (bypass) {
      pop_data_value =
          bb.Select(*is_empty_bool, {current_queue_tail, push_data});
    } else {
      pop_data_value = current_queue_tail;
    }

    if (register_pop) {
      bb.RegisterWrite(*pop_data_reg, pop_data_value,
                       /*load_enable=*/pop_valid_load_en, reset_port);
      pop_data_value = *pop_data_reg_read;
    }

    // We could always send the current data. If not r/v the data is ignored
    // anyway. Since this is testing might as well send mux in a zero if we
    // aren't ready.
    bb.OutputPort(FifoInstantiation::kPopDataPortName, pop_data_value);

    bb.RegisterWrite(buf_reg, next_buf_value_if_push_occurs,
                     /*load_enable=*/did_push_occur_bool, reset_port);
  }

  return bb.Build();
}
}  // namespace

absl::StatusOr<bool> MaybeMaterializeFifosPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  XLS_ASSIGN_OR_RETURN(BlockElaboration elab,
                       BlockElaboration::Elaborate(context.top_block()));
  struct MaterializedInstantiation {
    Block* instantiating_block;
    Instantiation* instantiation;
    Block* implementation = nullptr;
  };
  std::vector<MaterializedInstantiation> insts;
  for (Block* b : elab.blocks()) {
    for (xls::Instantiation* i : b->GetInstantiations()) {
      if (i->kind() == InstantiationKind::kFifo) {
        XLS_ASSIGN_OR_RETURN(FifoInstantiation * fifo,
                             i->AsFifoInstantiation());
        const bool should_materialize =
            fifo->data_type()->GetFlatBitCount() == 0
                ? options.codegen_options.nodata_fifo_module().empty()
                : options.codegen_options.fifo_module().empty();
        XLS_RETURN_IF_ERROR(fifo->fifo_config().Validate());
        if (should_materialize) {
          insts.push_back(MaterializedInstantiation{.instantiating_block = b,
                                                    .instantiation = i});
        }
      } else if (i->kind() == InstantiationKind::kDelayLine) {
        insts.push_back(MaterializedInstantiation{.instantiating_block = b,
                                                  .instantiation = i});
      }
    }
  }
  if (insts.empty()) {
    return false;
  }
  NameUniquer uniquer("___");
  // Intermediate list new blocks created.

  for (MaterializedInstantiation& inst : insts) {
    if (inst.instantiation->kind() == InstantiationKind::kFifo) {
      XLS_RET_CHECK(options.codegen_options.GetResetBehavior().has_value())
          << "Reset behavior must be set to materialize fifos";
      XLS_RET_CHECK(options.codegen_options.reset().has_value())
          << "Fifo materialization requires reset";
      XLS_RET_CHECK(options.codegen_options.reset().value().has_name())
          << "Fifo materialization requires reset name";
      std::string_view reset_name =
          options.codegen_options.reset().value().name();
      XLS_ASSIGN_OR_RETURN(
          inst.implementation,
          MaterializeFifo(uniquer, package,
                          inst.instantiation->AsFifoInstantiation().value(),
                          *options.codegen_options.GetResetBehavior(),
                          reset_name));
    } else {
      XLS_RET_CHECK_EQ(inst.instantiation->kind(),
                       InstantiationKind::kDelayLine);
      XLS_ASSIGN_OR_RETURN(
          inst.implementation,
          MaterializeDelayLine(
              inst.instantiating_block,
              inst.instantiation->AsDelayLineInstantiation().value()));
    }
  }
  for (const MaterializedInstantiation& inst : insts) {
    std::string old_reset_name;
    std::string name;
    if (inst.instantiation->kind() == InstantiationKind::kFifo) {
      name =
          absl::StrFormat("materialized_fifo_%s_", inst.instantiation->name());
      old_reset_name = FifoInstantiation::kResetPortName;
    } else {
      XLS_RET_CHECK_EQ(inst.instantiation->kind(),
                       InstantiationKind::kDelayLine);
      name = absl::StrFormat("materialized_delay_line_%s_",
                             inst.instantiation->name());
      old_reset_name = DelayLineInstantiation::kResetPortName;
    }

    // The name of the reset port of the materialized FIFO/delay-line is given
    // by the codegen options and hence might differ from the name of the reset
    // port of the original instantiation. This ensure the materialized block
    // behaves like any other require any special handling and acts like any
    // other and does not require special handling. This needs to be taken into
    // account when replacing the fifo/delay-line instantiation with
    // instantiation of the implementation.
    absl::flat_hash_map<std::string, std::string> port_renaming_rules;
    if (options.codegen_options.reset().has_value() &&
        options.codegen_options.reset()->name() != old_reset_name) {
      port_renaming_rules[old_reset_name] =
          options.codegen_options.reset()->name();
    }
    XLS_ASSIGN_OR_RETURN(xls::Instantiation * new_inst,
                         inst.instantiating_block->AddBlockInstantiation(
                             name, inst.implementation));
    XLS_RETURN_IF_ERROR(inst.instantiating_block->ReplaceInstantiationWith(
        inst.instantiation, new_inst, port_renaming_rules));
  }

  // Record all the elaboration registers added by this new block.
  XLS_ASSIGN_OR_RETURN(BlockElaboration new_elab,
                       BlockElaboration::Elaborate(context.top_block()));
  for (const MaterializedInstantiation& inst : insts) {
    for (BlockInstance* block_instance :
         new_elab.GetInstances(inst.implementation)) {
      for (Register* reg : inst.implementation->GetRegisters()) {
        context.inserted_registers()[absl::StrFormat(
            "%s%s", block_instance->RegisterPrefix(), reg->name())] =
            reg->type();
      }
    }
  }

  return true;
}

}  // namespace xls::verilog
