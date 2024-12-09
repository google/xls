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

#include "xls/codegen/materialize_fifos_pass.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
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
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls::verilog {
namespace {
absl::StatusOr<Block*> MaterializeFifo(NameUniquer& uniquer, Package* p,
                                       FifoInstantiation* inst,
                                       const xls::Reset& reset_behavior) {
  const FifoConfig& config = inst->fifo_config();
  const int64_t depth = config.depth();
  const bool bypass = config.bypass();
  const bool register_push = config.register_push_outputs();
  const bool register_pop = config.register_pop_outputs();
  Type* u1 = p->GetBitsType(1);
  Type* ty = inst->data_type();

  // Make sure there is one extra slot at least. Bad for QOR but makes impl
  // easier since full is always tail + size == head
  Type* buf_type = p->GetArrayType(depth + 1, ty);
  Type* ptr_type = p->GetBitsType(Bits::MinBitCountUnsigned(depth + 1));

  BlockBuilder bb(uniquer.GetSanitizedUniqueName(absl::StrFormat(
                      "fifo_for_depth_%d_ty_%s_%s%s%s", depth, ty->ToString(),
                      bypass ? "with_bypass" : "no_bypass",
                      register_pop ? "_register_pop" : "",
                      register_push ? "_register_push" : "")),
                  p);
  XLS_RETURN_IF_ERROR(bb.AddClockPort("clk"));
  BValue reset_port = bb.ResetPort(FifoInstantiation::kResetPortName);

  BValue one_lit = bb.Literal(UBits(1, ptr_type->GetFlatBitCount()));
  BValue depth_lit = bb.Literal(UBits(depth, ptr_type->GetFlatBitCount()));
  BValue long_buf_size_lit =
      bb.Literal(UBits(depth + 1, ptr_type->GetFlatBitCount() + 1),
                 SourceInfo(), "long_buf_size_lit");

  BValue push_valid = bb.InputPort(FifoInstantiation::kPushValidPortName, u1);
  BValue pop_ready_port =
      bb.InputPort(FifoInstantiation::kPopReadyPortName, u1);
  BValue push_data = bb.InputPort(FifoInstantiation::kPushDataPortName, ty);

  XLS_ASSIGN_OR_RETURN(
      Register * buf_reg,
      bb.block()->AddRegister(
          "buf", buf_type,
          xls::Reset{.reset_value = ZeroOfType(buf_type),
                     .asynchronous = reset_behavior.asynchronous,
                     .active_low = reset_behavior.active_low}));
  XLS_ASSIGN_OR_RETURN(
      Register * head_reg,
      bb.block()->AddRegister(
          "head", ptr_type,
          xls::Reset{.reset_value = ZeroOfType(ptr_type),
                     .asynchronous = reset_behavior.asynchronous,
                     .active_low = reset_behavior.active_low}));
  XLS_ASSIGN_OR_RETURN(
      Register * tail_reg,
      bb.block()->AddRegister(
          "tail", ptr_type,
          xls::Reset{.reset_value = ZeroOfType(ptr_type),
                     .asynchronous = reset_behavior.asynchronous,
                     .active_low = reset_behavior.active_low}));
  XLS_ASSIGN_OR_RETURN(
      Register * slots_reg,
      bb.block()->AddRegister(
          "slots", ptr_type,
          xls::Reset{.reset_value = ZeroOfType(ptr_type),
                     .asynchronous = reset_behavior.asynchronous,
                     .active_low = reset_behavior.active_low}));

  BValue buf = bb.RegisterRead(buf_reg);
  BValue head = bb.RegisterRead(head_reg, SourceInfo(), "head_ptr");
  BValue tail = bb.RegisterRead(tail_reg, SourceInfo(), "tail_ptr");
  BValue slots = bb.RegisterRead(slots_reg, SourceInfo(), "slots_ptr");

  // Only used if register_pop is true.
  std::optional<Register*> pop_valid_reg;
  std::optional<Register*> pop_data_reg;
  std::optional<BValue> pop_valid_reg_read;
  std::optional<BValue> pop_data_reg_read;
  std::optional<BValue> pop_valid_load_en;

  if (register_pop) {
    XLS_ASSIGN_OR_RETURN(
        pop_valid_reg,
        bb.block()->AddRegister(
            "pop_valid_reg", u1, /*reset=*/
            xls::Reset{.reset_value = Value::Bool(false),
                       .asynchronous = reset_behavior.asynchronous,
                       .active_low = reset_behavior.active_low}));
    XLS_ASSIGN_OR_RETURN(
        pop_data_reg,
        bb.block()->AddRegister(
            "pop_data_reg", ty, /*reset=*/
            xls::Reset{.reset_value = ZeroOfType(ty),
                       .asynchronous = reset_behavior.asynchronous,
                       .active_low = reset_behavior.active_low}));
    pop_valid_reg_read = bb.RegisterRead(*pop_valid_reg);
    depth_lit = bb.Add(depth_lit, bb.ZeroExtend(*pop_valid_reg_read,
                                                ptr_type->GetFlatBitCount()));
    pop_data_reg_read = bb.RegisterRead(*pop_data_reg);
    pop_valid_load_en = bb.Or(pop_ready_port, bb.Not(*pop_valid_reg_read));
  }

  BValue current_queue_tail = bb.ArrayIndex(buf, {tail});

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
  BValue next_buf_value_if_push_occurs = bb.ArrayUpdate(buf, push_data, {head});
  BValue pop_data_value;
  BValue did_pop_occur_bool;
  BValue did_push_occur_bool;
  if (bypass) {
    // NB No need to handle the 'full and bypass and both read & write case
    // specially since we have an extra slot to store those values in.
    BValue bypass_possible = bb.And(buf_pop_ready, push_valid);
    BValue is_pushable = bb.Or(can_do_push, bypass_possible);
    BValue is_popable = bb.Or(buf_not_empty_bool, bypass_possible);
    pop_valid = bb.Or(buf_not_empty_bool, push_valid);
    push_ready = bb.Or(can_do_push, buf_pop_ready);
    BValue is_empty_bool = bb.Eq(head, tail);
    BValue did_no_write_bypass_occur = bb.And(is_empty_bool, bypass_possible);
    BValue did_visible_push_occur_bool = bb.And(is_pushable, push_valid);
    if (register_pop) {
      buf_pop_ready = bb.And(buf_pop_ready, pop_valid);
    }
    BValue did_visible_pop_occur_bool = bb.And(is_popable, buf_pop_ready);
    pop_data_value = bb.Select(is_empty_bool, {current_queue_tail, push_data});
    did_pop_occur_bool =
        bb.And(did_visible_pop_occur_bool, bb.Not(did_no_write_bypass_occur),
               SourceInfo(), "did_pop_occur");
    did_push_occur_bool =
        bb.And(did_visible_push_occur_bool, bb.Not(did_no_write_bypass_occur));
  } else {
    push_ready = can_do_push;
    pop_valid = buf_not_empty_bool;
    pop_data_value = current_queue_tail;
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
    bb.RegisterWrite(*pop_data_reg, pop_data_value,
                     /*load_enable=*/pop_valid_load_en, reset_port);

    pop_valid = *pop_valid_reg_read;
    pop_data_value = *pop_data_reg_read;
  }
  bb.OutputPort(FifoInstantiation::kPushReadyPortName, push_ready);

  // Empty is head == tail so ne means there's something to pop.
  bb.OutputPort(FifoInstantiation::kPopValidPortName, pop_valid);

  // We could always send the current data. If not r/v the data is ignored
  // anyway. Since this is testing might as well send mux in a zero if we
  // aren't ready.
  bb.OutputPort(FifoInstantiation::kPopDataPortName, pop_data_value);

  BValue pushed = bb.And(push_ready, push_valid, SourceInfo(), "pushed");
  BValue popped = bb.And(pop_ready_port, pop_valid, SourceInfo(), "popped");
  BValue slots_next = bb.Select(
      pushed, {bb.Select(popped, {slots, bb.Subtract(slots, one_lit)}),
               bb.Select(popped, {bb.Add(slots, one_lit), slots})});

  bb.RegisterWrite(buf_reg, next_buf_value_if_push_occurs,
                   /*load_enable=*/did_push_occur_bool, reset_port);
  bb.RegisterWrite(head_reg, next_head_value_if_push_occurs,
                   /*load_enable=*/did_push_occur_bool, reset_port);
  bb.RegisterWrite(tail_reg, next_tail_value_if_pop_occurs,
                   /*load_enable=*/did_pop_occur_bool, reset_port);
  bb.RegisterWrite(slots_reg, slots_next, /*load_enable=*/std::nullopt,
                   reset_port);

  return bb.Build();
}
}  // namespace

absl::StatusOr<bool> MaterializeFifosPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  XLS_ASSIGN_OR_RETURN(BlockElaboration elab,
                       BlockElaboration::Elaborate(unit->top_block));
  std::vector<FifoInstantiation*> insts;
  for (Block* b : elab.blocks()) {
    for (xls::Instantiation* i : b->GetInstantiations()) {
      if (i->kind() == InstantiationKind::kFifo) {
        XLS_ASSIGN_OR_RETURN(FifoInstantiation * fifo,
                             i->AsFifoInstantiation());
        if (fifo->fifo_config().depth() == 1 &&
            fifo->fifo_config().register_pop_outputs()) {
          return absl::InvalidArgumentError(
              "Cannot materialize fifo with register_pop_outputs and depth 1.");
        }
        insts.push_back(fifo);
      }
    }
  }
  if (insts.empty()) {
    return false;
  }
  NameUniquer uniquer("___");
  // Intermediate list new blocks created.

  absl::flat_hash_map<xls::Instantiation*, Block*> impls;
  for (FifoInstantiation* f : insts) {
    XLS_RET_CHECK(options.codegen_options.ResetBehavior().has_value())
        << "Reset behavior must be set to materialize fifos";
    XLS_ASSIGN_OR_RETURN(
        impls[f], MaterializeFifo(uniquer, unit->package, f,
                                  *options.codegen_options.ResetBehavior()));
  }

  std::vector<Block*> saved_blocks(elab.blocks().begin(), elab.blocks().end());
  for (Block* b : saved_blocks) {
    std::vector<xls::Instantiation*> saved_instantiations(
        b->GetInstantiations().begin(), b->GetInstantiations().end());
    for (xls::Instantiation* i : saved_instantiations) {
      if (!impls.contains(i)) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          xls::Instantiation * new_inst,
          b->AddBlockInstantiation(
              absl::StrFormat("materialized_fifo_%s_", i->name()),
              impls.at(i)));
      XLS_RETURN_IF_ERROR(b->ReplaceInstantiationWith(i, new_inst));
    }
  }

  // Record all the elaboration registers added by this new block.
  XLS_ASSIGN_OR_RETURN(BlockElaboration new_elab,
                       BlockElaboration::Elaborate(unit->top_block));
  for (const auto& [i, blk] : impls) {
    for (BlockInstance* inst : new_elab.GetInstances(blk)) {
      for (Register* reg : blk->GetRegisters()) {
        results->inserted_registers[absl::StrFormat(
            "%s%s", inst->RegisterPrefix(), reg->name())] = reg->type();
      }
    }
  }
  return true;
}

absl::StatusOr<bool> MaybeMaterializeInternalFifoPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  if (options.codegen_options.materialize_internal_fifos()) {
    return materialize_.Run(unit, options, results);
  }
  return false;
}

}  // namespace xls::verilog
