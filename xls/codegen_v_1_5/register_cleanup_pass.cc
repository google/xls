// Copyright 2025 The XLS Authors
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

#include "xls/codegen_v_1_5/register_cleanup_pass.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/transitive_closure.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"

namespace xls::codegen {

namespace {

absl::StatusOr<std::vector<RegisterWrite*>> GetRegisterWrites(Block* block,
                                                              Register* reg) {
  XLS_ASSIGN_OR_RETURN(absl::Span<RegisterWrite* const> writes,
                       block->GetRegisterWrites(reg));
  return std::vector<RegisterWrite*>(writes.begin(), writes.end());
}

}  // namespace

absl::StatusOr<bool> RegisterCleanupPass::RemoveTrivialLoadEnables(
    Block* block, QueryEngine& query_engine) const {
  bool changed = false;
  for (Register* reg : block->GetRegisters()) {
    XLS_ASSIGN_OR_RETURN(std::vector<RegisterWrite*> writes,
                         GetRegisterWrites(block, reg));
    for (RegisterWrite* write : writes) {
      if (write->load_enable().has_value() &&
          query_engine.IsAllOnes(*write->load_enable())) {
        XLS_RETURN_IF_ERROR(
            write
                ->ReplaceUsesWithNew<RegisterWrite>(write->data(), std::nullopt,
                                                    write->reset(), reg)
                .status());
        XLS_RETURN_IF_ERROR(block->RemoveNode(write));
        changed = true;
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> RegisterCleanupPass::RemoveImpossibleWrites(
    Block* block, QueryEngine& query_engine) const {
  std::vector<Register*> registers(block->GetRegisters().begin(),
                                   block->GetRegisters().end());
  for (Register* reg : registers) {
    XLS_ASSIGN_OR_RETURN(std::vector<RegisterWrite*> writes,
                         GetRegisterWrites(block, reg));
    for (RegisterWrite* write : writes) {
      bool write_disabled = write->load_enable().has_value() &&
                            query_engine.IsAllZeros(*write->load_enable());
      if (write_disabled) {
        XLS_RETURN_IF_ERROR(
            write->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
        XLS_RETURN_IF_ERROR(block->RemoveNode(write));
      }
    }

    XLS_ASSIGN_OR_RETURN(writes, GetRegisterWrites(block, reg));
    if (absl::c_all_of(writes, [&](RegisterWrite* write) {
          return reg->reset_value().has_value() &&
                 query_engine.KnownValue(write->data()) == *reg->reset_value();
        })) {
      // Nothing can ever write a new value to this register, so it's equivalent
      // to its reset value. We can remove the register & all ops.
      XLS_ASSIGN_OR_RETURN(RegisterRead * read, block->GetRegisterRead(reg));
      XLS_RETURN_IF_ERROR(
          read->ReplaceUsesWithNew<Literal>(
                  reg->reset_value().value_or(ZeroOfType(reg->type())))
              .status());
      XLS_RETURN_IF_ERROR(block->RemoveNode(read));
      for (RegisterWrite* write : writes) {
        XLS_RETURN_IF_ERROR(
            write->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
        XLS_RETURN_IF_ERROR(block->RemoveNode(write));
      }
      XLS_RETURN_IF_ERROR(block->RemoveRegister(reg));
    }
  }
  return false;
}

absl::StatusOr<bool> RegisterCleanupPass::RemoveUnreadRegisters(
    Block* block, QueryEngine& query_engine) const {
  absl::flat_hash_map<Register*, absl::flat_hash_set<Register*>>
      can_receive_value_from;
  absl::flat_hash_set<Register*> directly_visible_registers;
  NodeBackwardDependencyAnalysis nda;
  XLS_RETURN_IF_ERROR(nda.Attach(block).status());
  for (Register* reg : block->GetRegisters()) {
    XLS_ASSIGN_OR_RETURN(RegisterRead * read, block->GetRegisterRead(reg));
    for (Node* user : nda.NodesDependingOn(read)) {
      if (user == read) {
        continue;
      }

      if (user->Is<RegisterWrite>()) {
        can_receive_value_from[reg].insert(
            user->As<RegisterWrite>()->GetRegister());
      } else if (OpIsSideEffecting(user->op())) {
        directly_visible_registers.insert(reg);
      }
    }
  }

  absl::flat_hash_map<Register*, absl::flat_hash_set<Register*>>
      transitive_receivers = TransitiveClosure(can_receive_value_from);
  std::vector<Register*> unread_registers;
  for (Register* reg : block->GetRegisters()) {
    if (!directly_visible_registers.contains(reg) &&
        absl::c_none_of(transitive_receivers[reg], [&](Register* user) {
          return directly_visible_registers.contains(user);
        })) {
      unread_registers.push_back(reg);
    }
  }

  for (Register* reg : unread_registers) {
    XLS_ASSIGN_OR_RETURN(RegisterRead * read, block->GetRegisterRead(reg));
    XLS_ASSIGN_OR_RETURN(absl::Span<RegisterWrite* const> writes,
                         block->GetRegisterWrites(reg));
    if (!read->IsDead()) {
      XLS_RETURN_IF_ERROR(
          read->ReplaceUsesWithNew<Literal>(
                  reg->reset_value().value_or(ZeroOfType(reg->type())))
              .status());
    }
    XLS_RETURN_IF_ERROR(block->RemoveNode(read));
    for (RegisterWrite* write : writes) {
      if (!write->IsDead()) {
        XLS_RETURN_IF_ERROR(
            write->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      }
      XLS_RETURN_IF_ERROR(block->RemoveNode(write));
    }
    XLS_RETURN_IF_ERROR(block->RemoveRegister(reg));
  }
  return !unread_registers.empty();
}

absl::StatusOr<bool> RegisterCleanupPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    PartialInfoQueryEngine query_engine;
    XLS_RETURN_IF_ERROR(query_engine.Populate(block.get()).status());

    XLS_ASSIGN_OR_RETURN(bool changed_load_enables,
                         RemoveTrivialLoadEnables(block.get(), query_engine));
    changed |= changed_load_enables;

    XLS_ASSIGN_OR_RETURN(bool changed_writes,
                         RemoveImpossibleWrites(block.get(), query_engine));
    changed |= changed_writes;

    XLS_ASSIGN_OR_RETURN(bool changed_registers,
                         RemoveUnreadRegisters(block.get(), query_engine));
    changed |= changed_registers;
  }
  return changed;
}

}  // namespace xls::codegen
