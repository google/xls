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

#include "xls/codegen_v_1_5/merge_registers_pass.h"

#include <algorithm>
#include <compare>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/linked_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

struct RegisterWriteSignature {
  Node* data;
  std::optional<Node*> load_enable;
  std::optional<Node*> reset;

  bool operator==(const RegisterWriteSignature& other) const = default;

  std::strong_ordering operator<=>(const RegisterWriteSignature& other) const {
    // This is an arbitrary ordering, but it lets us put writes in a canonical
    // order, which is all we need.
    //
    // For each field, a missing value always compares as "less than" a present
    // value.
    if (auto cmp = data->id() <=> other.data->id();
        cmp != std::strong_ordering::equal) {
      return cmp;
    }
    if (auto cmp = load_enable.has_value() <=> other.load_enable.has_value();
        cmp != std::strong_ordering::equal) {
      return cmp;
    }
    if (load_enable.has_value()) {
      if (auto cmp = (*load_enable)->id() <=> (*other.load_enable)->id();
          cmp != std::strong_ordering::equal) {
        return cmp;
      }
    }
    if (auto cmp = reset.has_value() <=> other.reset.has_value();
        cmp != std::strong_ordering::equal) {
      return cmp;
    }
    if (reset.has_value()) {
      if (auto cmp = (*reset)->id() <=> (*other.reset)->id();
          cmp != std::strong_ordering::equal) {
        return cmp;
      }
    }
    return std::strong_ordering::equal;
  }

  template <typename H>
  friend H AbslHashValue(H h, const RegisterWriteSignature& s) {
    return H::combine(std::move(h), s.data, s.load_enable, s.reset);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RegisterWriteSignature& s) {
    absl::Format(
        &sink, "{data: %s, load_enable: %s, reset: %s}", s.data->ToString(),
        s.load_enable.has_value() ? (*s.load_enable)->ToString() : "nullopt",
        s.reset.has_value() ? (*s.reset)->ToString() : "nullopt");
  }
};

struct RegisterSignature {
  Type* type;
  std::optional<Value> reset_value;
  std::vector<RegisterWriteSignature> writes;

  bool operator==(const RegisterSignature& other) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const RegisterSignature& s) {
    return H::combine(std::move(h), s.type, s.reset_value, s.writes);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RegisterSignature& s) {
    absl::Format(
        &sink, "{type: %s, reset_value: %s, writes: %s}", s.type->ToString(),
        s.reset_value.has_value() ? s.reset_value->ToString() : "nullopt",
        absl::StrJoin(s.writes, ", "));
  }
};

absl::StatusOr<RegisterSignature> GetRegisterSignature(Register* reg,
                                                       Block* block) {
  XLS_ASSIGN_OR_RETURN(absl::Span<RegisterWrite* const> writes,
                       block->GetRegisterWrites(reg));

  std::vector<RegisterWriteSignature> write_sigs;
  write_sigs.reserve(writes.size());
  for (RegisterWrite* write : writes) {
    write_sigs.push_back(RegisterWriteSignature{
        .data = write->data(),
        .load_enable = write->load_enable(),
        .reset = write->reset(),
    });
  }

  // Put the writes in a canonical order, for easy comparison.
  absl::c_sort(write_sigs);

  return RegisterSignature{
      .type = reg->type(),
      .reset_value = reg->reset_value(),
      .writes = std::move(write_sigs),
  };
}

absl::Status MergeRegisters(Register* authoritative_reg, Register* dupe_reg,
                            Block* block) {
  auto dupe_read_status = block->GetRegisterRead(dupe_reg);
  if (dupe_read_status.ok()) {
    RegisterRead* dupe_read = *dupe_read_status;
    auto auth_read_status = block->GetRegisterRead(authoritative_reg);
    RegisterRead* auth_read = nullptr;
    if (auth_read_status.ok()) {
      auth_read = *auth_read_status;
    } else {
      XLS_ASSIGN_OR_RETURN(
          auth_read, block->MakeNodeWithName<RegisterRead>(
                         dupe_read->loc(), authoritative_reg,
                         absl::StrCat(authoritative_reg->name(), "_read")));
    }
    XLS_RETURN_IF_ERROR(dupe_read->ReplaceUsesWith(auth_read));
    XLS_RETURN_IF_ERROR(block->RemoveNode(dupe_read));
  }

  XLS_ASSIGN_OR_RETURN(absl::Span<RegisterWrite* const> dupe_writes,
                       block->GetRegisterWrites(dupe_reg));
  std::vector<RegisterWrite*> writes_to_remove(dupe_writes.begin(),
                                               dupe_writes.end());
  for (RegisterWrite* write : writes_to_remove) {
    XLS_RETURN_IF_ERROR(block->RemoveNode(write));
  }

  return block->RemoveRegister(dupe_reg);
}

absl::StatusOr<bool> MergeRegistersInBlock(Block* block) {
  bool changed = false;

  absl::linked_hash_map<RegisterSignature, std::vector<Register*>>
      equivalent_registers;
  equivalent_registers.reserve(block->GetRegisters().size());
  for (Register* reg : block->GetRegisters()) {
    XLS_ASSIGN_OR_RETURN(RegisterSignature sig,
                         GetRegisterSignature(reg, block));
    auto& equivalents = equivalent_registers[sig];
    equivalents.push_back(reg);
  }

  for (const auto& [sig, equivalents] : equivalent_registers) {
    Register* authoritative_reg = equivalents[0];
    for (Register* dupe_reg : absl::MakeConstSpan(equivalents).subspan(1)) {
      XLS_RETURN_IF_ERROR(MergeRegisters(authoritative_reg, dupe_reg, block));
      changed = true;
    }
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> MergeRegistersPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results, BlockConversionContext& context) const {
  if (options.codegen_options.register_merge_strategy() ==
      verilog::CodegenOptions::RegisterMergeStrategy::kDontMerge) {
    return false;
  }

  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    while (true) {
      XLS_ASSIGN_OR_RETURN(bool merged, MergeRegistersInBlock(block.get()));
      if (!merged) {
        break;
      }
      changed |= merged;
    }
  }
  return changed;
}

}  // namespace xls::codegen
