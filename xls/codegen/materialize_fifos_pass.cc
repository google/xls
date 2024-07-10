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
#include "xls/ir/value_utils.h"

namespace xls::verilog {
namespace {
absl::StatusOr<Block*> MaterializeFifo(NameUniquer& uniquer, Package* p,
                                       FifoInstantiation* inst,
                                       const xls::Reset& reset_behavior) {
  const FifoConfig& config = inst->fifo_config();
  if (config.bypass()) {
    return absl::UnimplementedError("Bypass Not yet supported");
  }
  if (config.register_push_outputs()) {
    return absl::UnimplementedError("PushReg Not yet supported");
  }
  if (config.register_pop_outputs()) {
    return absl::UnimplementedError("PopReg Not yet supported");
  }
  int64_t depth = config.depth();
  Type* u1 = p->GetBitsType(1);
  Type* ty = inst->data_type();
  BlockBuilder bb(uniquer.GetSanitizedUniqueName(absl::StrFormat(
                      "fifo_for_depth_%d_ty_%s", depth, ty->ToString())),
                  p);
  XLS_RETURN_IF_ERROR(bb.AddClockPort("clk"));
  XLS_ASSIGN_OR_RETURN(
      InputPort * reset_port,
      bb.block()->AddResetPort(FifoInstantiation::kResetPortName));
  BValue reset_port_bvalue(reset_port, &bb);

  // Make sure there is one extra slot at least. Bad for QOR but makes impl
  // easier since full is always tail + size == head
  Type* buf_type = p->GetArrayType(depth + 1, ty);
  Type* ptr_type = p->GetBitsType(Bits::MinBitCountUnsigned(depth + 1));
  XLS_ASSIGN_OR_RETURN(Register * buf_reg,
                       bb.block()->AddRegister("buf", buf_type));
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
  BValue one_lit = bb.Literal(UBits(1, ptr_type->GetFlatBitCount()));
  BValue depth_lit = bb.Literal(UBits(depth, ptr_type->GetFlatBitCount()));
  BValue long_buf_size_lit =
      bb.Literal(UBits(depth + 1, ptr_type->GetFlatBitCount() + 1),
                 SourceInfo(), "long_buf_size_lit");

  BValue push_valid = bb.InputPort(FifoInstantiation::kPushValidPortName, u1);
  BValue pop_ready = bb.InputPort(FifoInstantiation::kPopReadyPortName, u1);
  BValue push_data = bb.InputPort(FifoInstantiation::kPushDataPortName, ty);

  BValue buf = bb.RegisterRead(buf_reg);
  BValue head = bb.RegisterRead(head_reg, SourceInfo(), "head_ptr");
  BValue tail = bb.RegisterRead(tail_reg, SourceInfo(), "tail_ptr");

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

  // Output the current state.
  BValue is_full_bool =
      bb.Eq(head, add_mod_buf_size(tail, depth_lit, "tail_plus_depth"));
  BValue not_full_bool = bb.Not(is_full_bool);
  BValue can_do_push =
      bb.Or(not_full_bool, pop_ready, SourceInfo(), "can_do_push");
  // Ready to take things if we are not full.
  bb.OutputPort(FifoInstantiation::kPushReadyPortName, can_do_push);

  // Empty is head == tail so ne means there's something to pop.
  BValue not_empty_bool = bb.Ne(head, tail);
  bb.OutputPort(FifoInstantiation::kPopValidPortName, not_empty_bool);

  BValue next_tail_value_if_pop_occurs =
      add_mod_buf_size(tail, one_lit, "next_tail_if_pop");
  BValue next_head_value_if_push_occurs =
      add_mod_buf_size(head, one_lit, "next_head_if_push");
  // NB we don't bother clearing data after a pop.
  BValue next_buf_value_if_push_occurs = bb.ArrayUpdate(buf, push_data, {head});

  BValue did_push_occur_bool = bb.And(can_do_push, push_valid);
  BValue did_pop_occur_bool = bb.And(not_empty_bool, pop_ready);

  bb.OutputPort(FifoInstantiation::kPopDataPortName, current_queue_tail);

  bb.RegisterWrite(buf_reg, next_buf_value_if_push_occurs,
                   /*load_enable=*/did_push_occur_bool);
  bb.RegisterWrite(head_reg, next_head_value_if_push_occurs,
                   /*load_enable=*/did_push_occur_bool, reset_port_bvalue);
  bb.RegisterWrite(tail_reg, next_tail_value_if_pop_occurs,
                   /*load_enable=*/did_pop_occur_bool, reset_port_bvalue);

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
        insts.push_back(fifo);
      }
    }
  }
  if (insts.empty()) {
    return false;
  }
  NameUniquer uniquer("___");

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
  return true;
}
}  // namespace xls::verilog
