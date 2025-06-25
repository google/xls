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

#include "xls/ir/package.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/call_graph.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/transform_metrics.pb.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

Package::Package(std::string_view name) : name_(name) {}

Package::~Package() = default;

std::optional<FunctionBase*> Package::GetTop() const { return top_; }

absl::Status Package::SetTop(std::optional<FunctionBase*> top) {
  if (top.has_value() && top.value()->package() != this) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot set the top entity of the package: the top entity %s does not "
        "belong to the package.",
        top.value()->name()));
  }
  top_ = top;
  return absl::OkStatus();
}

absl::Status Package::SetTopByName(std::string_view top_name) {
  XLS_ASSIGN_OR_RETURN(FunctionBase * top, GetFunctionBaseByName(top_name));
  return SetTop(top);
}

absl::StatusOr<Function*> Package::GetTopAsFunction() const {
  std::optional<FunctionBase*> top = GetTop();
  if (!top.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Top entity not set for package: %s.", name_));
  }
  if (!top.value()->IsFunction()) {
    return absl::InternalError(absl::StrFormat(
        "Top entity is not a function for package: %s.", name_));
  }
  return top.value()->AsFunctionOrDie();
}

absl::StatusOr<Proc*> Package::GetTopAsProc() const {
  std::optional<FunctionBase*> top = GetTop();
  if (!top.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Top entity not set for package: %s.", name_));
  }
  if (!top.value()->IsProc()) {
    return absl::InternalError(
        absl::StrFormat("Top entity is not a proc for package: %s.", name_));
  }
  return top.value()->AsProcOrDie();
}

absl::StatusOr<Block*> Package::GetTopAsBlock() const {
  std::optional<FunctionBase*> top = GetTop();
  if (!top.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Top entity not set for package: %s.", name_));
  }
  if (!top.value()->IsBlock()) {
    return absl::InternalError(
        absl::StrFormat("Top entity is not a block for package: %s.", name_));
  }
  return top.value()->AsBlockOrDie();
}

absl::StatusOr<FunctionBase*> Package::GetFunctionBaseByName(
    std::string_view name) const {
  std::vector<FunctionBase*> matches;
  for (auto& function : functions()) {
    if (function->name() == name) {
      matches.push_back(function.get());
    }
  }
  for (auto& proc : procs()) {
    if (proc->name() == name) {
      matches.push_back(proc.get());
    }
  }
  for (auto& block : blocks()) {
    if (block->name() == name) {
      matches.push_back(block.get());
    }
  }
  if (matches.empty()) {
    std::vector<FunctionBase*> fbs = GetFunctionBases();
    std::string available =
        absl::StrJoin(fbs.begin(), fbs.end(), ", ",
                      [](std::string* out, const FunctionBase* fb) {
                        absl::StrAppend(out, "\"", fb->name(), "\"");
                      });
    return absl::NotFoundError(
        absl::StrFormat("Could not find top for this package; "
                        "tried: [\"%s\"]; available: %s",
                        name, available));
  }
  if (matches.size() == 1) {
    return matches.front();
  }
  return absl::NotFoundError(
      absl::StrFormat("More than one instance with name: %s", name));
}

Function* Package::AddFunction(std::unique_ptr<Function> f) {
  functions_.push_back(std::move(f));
  return functions_.back().get();
}

Proc* Package::AddProc(std::unique_ptr<Proc> proc) {
  procs_.push_back(std::move(proc));
  return procs_.back().get();
}

Block* Package::AddBlock(std::unique_ptr<Block> block) {
  blocks_.push_back(std::move(block));
  return blocks_.back().get();
}

// Private helpers for Package::ImportFromPackage().
namespace {
// Helper class that tracks names in a package and resolves name collisions.
class NameCollisionResolver {
 public:
  explicit NameCollisionResolver(absl::flat_hash_set<std::string_view> names)
      : names_(std::move(names)), name_updates_({}) {}

  const absl::flat_hash_map<std::string, std::string>& name_updates() const {
    return name_updates_;
  }
  const absl::flat_hash_set<std::string_view>& names() const { return names_; }

  bool Collides(std::string_view name) { return names_.contains(name); }

  std::string ResolveName(std::string_view old_name) {
    if (!Collides(old_name)) {
      return std::string(old_name);
    }
    std::string new_name;
    int suffix = 1;
    do {
      new_name = absl::StrCat(old_name, "_", suffix);
      ++suffix;
    } while (Collides(new_name));
    name_updates_[old_name] = new_name;
    std::string& new_name_ref = storage_.emplace_back(new_name);
    names_.insert(new_name_ref);
    return new_name;
  }

 private:
  // Set of every name of a function, proc, or block. We'll need to check
  // for collisions and resolve them.
  absl::flat_hash_set<std::string_view> names_;
  absl::flat_hash_map<std::string, std::string> name_updates_;
  // Storage for string_views inside names_ that have been created in
  // ResolveName.
  std::deque<std::string> storage_;
};

// Get a set of all names defined within a package.
absl::flat_hash_set<std::string_view> AllPackageNames(const Package& package) {
  absl::flat_hash_set<std::string_view> names;

  for (auto& function : package.functions()) {
    names.insert(function->name());
  }
  for (auto& channel : package.channels()) {
    names.insert(channel->name());
  }
  for (auto& proc : package.procs()) {
    names.insert(proc->name());
  }
  for (auto& block : package.blocks()) {
    names.insert(block->name());
  }

  return names;
}

// Adds channels from other_package to this_package, potentially changing the
// channel id. Returns channel name mapping from old name -> new name.
absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
AddChannelsFromPackage(Package* this_package, const Package* other_package,
                       NameCollisionResolver* name_resolver) {
  absl::flat_hash_map<std::string, std::string> channel_updates;
  // Channels can collide in two ways: by name, and by id. First we resolve name
  // collisions, and then we call the various Create*Channel() functions, which
  // will give a new channel id. We keep track of this new id to update
  // references to it later.
  for (const auto channel : other_package->channels()) {
    std::string channel_name = name_resolver->ResolveName(channel->name());
    XLS_ASSIGN_OR_RETURN(Channel * new_channel,
                         this_package->CloneChannel(channel, channel_name));
    channel_updates[channel->name()] = new_channel->name();
  }
  return channel_updates;
}

// Add FunctionBases (function, proc, and block) from other_package to
// this_package. Assumes channels have already been added.
absl::StatusOr<absl::flat_hash_map<const FunctionBase*, FunctionBase*>>
AddFunctionBasesFromPackage(
    Package* this_package, const Package* other_package,
    NameCollisionResolver* name_resolver,
    const absl::flat_hash_map<std::string, std::string>& channel_remapping) {
  std::vector<FunctionBase*> other_function_bases =
      other_package->GetFunctionBases();

  absl::flat_hash_map<const FunctionBase*, FunctionBase*>
      function_base_remapping;
  function_base_remapping.reserve(other_function_bases.size());

  // Cloning functions takes a map from const Function*->Function* instead of
  // FunctionBase. Keeping a separate map up to date in parallel is not ideal,
  // but better than making a new subset copy for every function.
  absl::flat_hash_map<const Function*, Function*> function_remapping;
  function_remapping.reserve(other_package->functions().size());

  for (auto& caller : other_function_bases) {
    if (function_base_remapping.contains(caller)) {
      continue;
    }
    // GetDependentFunctions() returns a DFS, so no need for more bookkeeping to
    // make sure we are cloning in dependency order.
    for (FunctionBase* callee : GetDependentFunctions(caller)) {
      if (function_base_remapping.contains(callee)) {
        continue;
      }
      // If needed, find a new name for the current callee and clone it into the
      // current package.
      std::string new_name = name_resolver->ResolveName(callee->name());
      if (callee->IsFunction()) {
        XLS_ASSIGN_OR_RETURN(Function * new_callee,
                             callee->AsFunctionOrDie()->Clone(
                                 new_name, this_package, function_remapping));
        function_base_remapping[callee] = new_callee;
        function_remapping[callee->AsFunctionOrDie()] = new_callee;
      } else if (callee->IsProc()) {
        XLS_ASSIGN_OR_RETURN(Proc * new_callee,
                             callee->AsProcOrDie()->Clone(
                                 new_name, this_package, channel_remapping,
                                 function_base_remapping));
        function_base_remapping[callee] = new_callee;
      } else if (callee->IsBlock()) {
        XLS_ASSIGN_OR_RETURN(Block * new_block, callee->AsBlockOrDie()->Clone(
                                                    new_name, this_package));
        function_base_remapping[callee] = new_block;
      } else {
        return absl::InvalidArgumentError(absl::StrFormat(
            "FunctionBase %s was not a function, proc, or block.",
            callee->name()));
      }
    }
  }

  return function_base_remapping;
}
}  // namespace

absl::StatusOr<Package::PackageMergeResult> Package::ImportFromPackage(
    const Package* other) {
  // Helper that keeps track of old -> new name mapping, resolving collisions if
  // needed.
  NameCollisionResolver name_resolver(AllPackageNames(*this));

  // First, merge channels.
  // Returns a mapping of channel ids from old name -> new name
  XLS_ASSIGN_OR_RETURN(auto channel_updates,
                       AddChannelsFromPackage(this, other, &name_resolver));

  // Next, merge in functions, procs, and blocks.
  XLS_ASSIGN_OR_RETURN(auto call_mapping,
                       AddFunctionBasesFromPackage(this, other, &name_resolver,
                                                   channel_updates));
  return Package::PackageMergeResult{
      .name_updates = name_resolver.name_updates(),
      .channel_updates = std::move(channel_updates)};
}

absl::StatusOr<Function*> Package::GetFunction(
    std::string_view func_name) const {
  for (auto& f : functions_) {
    if (f->name() == func_name) {
      return f.get();
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "Package does not have a function with name: \"%s\"; available: [%s]",
      func_name,
      absl::StrJoin(functions_, ", ",
                    [](std::string* out, const std::unique_ptr<Function>& f) {
                      absl::StrAppend(out, f->name());
                    })));
}

absl::StatusOr<Proc*> Package::GetProc(std::string_view proc_name) const {
  for (auto& p : procs_) {
    if (p->name() == proc_name) {
      return p.get();
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "Package does not have a proc with name: \"%s\"; available: [%s]",
      proc_name,
      absl::StrJoin(procs_, ", ",
                    [](std::string* out, const std::unique_ptr<Proc>& p) {
                      absl::StrAppend(out, p->name());
                    })));
}

absl::StatusOr<Block*> Package::GetBlock(std::string_view block_name) const {
  for (auto& block : blocks_) {
    if (block->name() == block_name) {
      return block.get();
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "Package does not have a block with name: \"%s\"; available: [%s]",
      block_name,
      absl::StrJoin(blocks_, ", ",
                    [](std::string* out, const std::unique_ptr<Block>& block) {
                      absl::StrAppend(out, block->name());
                    })));
}

std::optional<Function*> Package::TryGetFunction(
    std::string_view func_name) const {
  for (auto& f : functions_) {
    if (f->name() == func_name) {
      return f.get();
    }
  }
  return std::nullopt;
}

std::optional<Proc*> Package::TryGetProc(std::string_view proc_name) const {
  for (auto& p : procs_) {
    if (p->name() == proc_name) {
      return p.get();
    }
  }
  return std::nullopt;
}

std::optional<Block*> Package::TryGetBlock(std::string_view block_name) const {
  for (auto& block : blocks_) {
    if (block->name() == block_name) {
      return block.get();
    }
  }
  return std::nullopt;
}

std::vector<FunctionBase*> Package::GetFunctionBases() const {
  std::vector<FunctionBase*> result;
  for (auto& function : functions()) {
    result.push_back(function.get());
  }
  for (auto& proc : procs()) {
    result.push_back(proc.get());
  }
  for (auto& block : blocks()) {
    result.push_back(block.get());
  }
  return result;
}

absl::Status Package::RemoveFunctionBase(FunctionBase* function_base) {
  if (function_base->IsFunction()) {
    return RemoveFunction(function_base->AsFunctionOrDie());
  }
  if (function_base->IsProc()) {
    return RemoveProc(function_base->AsProcOrDie());
  }
  return RemoveBlock(function_base->AsBlockOrDie());
}

absl::Status Package::RemoveFunction(Function* function) {
  if (top_.has_value() && top_.value() == function) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot remove function: %s. The function is the top entity.",
        function->name()));
  }
  auto it = std::remove_if(
      functions_.begin(), functions_.end(),
      [&](const std::unique_ptr<Function>& f) { return f.get() == function; });
  if (it == functions_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "`%s` is not a function in package `%s`", function->name(), name()));
  }
  functions_.erase(it, functions_.end());
  return absl::OkStatus();
}

absl::Status Package::RemoveProc(Proc* proc) {
  if (top_.has_value() && top_.value() == proc) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot remove proc: %s. The proc is the top entity.", proc->name()));
  }
  auto it = std::remove_if(
      procs_.begin(), procs_.end(),
      [&](const std::unique_ptr<Proc>& f) { return f.get() == proc; });
  if (it == procs_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "`%s` is not a proc in package `%s`", proc->name(), name()));
  }
  procs_.erase(it, procs_.end());
  return absl::OkStatus();
}

absl::Status Package::RemoveBlock(Block* block) {
  if (top_.has_value() && top_.value() == block) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot remove block: %s. The block is the top entity.",
                        block->name()));
  }
  auto it = std::remove_if(
      blocks_.begin(), blocks_.end(),
      [&](const std::unique_ptr<Block>& f) { return f.get() == block; });
  if (it == blocks_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "`%s` is not a block in package `%s`", block->name(), name()));
  }
  blocks_.erase(it, blocks_.end());
  return absl::OkStatus();
}

SourceLocation Package::AddSourceLocation(std::string_view filename,
                                          Lineno lineno, Colno colno) {
  Fileno this_fileno = GetOrCreateFileno(filename);
  return SourceLocation(this_fileno, lineno, colno);
}

std::string Package::SourceLocationToString(const SourceLocation& loc) {
  const std::string unknown = "UNKNOWN";
  std::string_view filename =
      fileno_to_filename_.find(loc.fileno()) != fileno_to_filename_.end()
          ? fileno_to_filename_.at(loc.fileno())
          : unknown;
  return absl::StrFormat("%s:%d", filename, loc.lineno().value());
}

Fileno Package::GetOrCreateFileno(std::string_view filename) {
  // Attempt to add a new fileno/filename pair to the map.
  if (auto it = filename_to_fileno_.find(std::string(filename));
      it != filename_to_fileno_.end()) {
    return it->second;
  }
  Fileno this_fileno =
      maximum_fileno_.has_value()
          ? Fileno(static_cast<int32_t>(maximum_fileno_.value()) + 1)
          : Fileno(0);
  filename_to_fileno_.emplace(std::string(filename), this_fileno);
  fileno_to_filename_.emplace(this_fileno, std::string(filename));
  maximum_fileno_ = this_fileno;

  return this_fileno;
}

void Package::SetFileno(Fileno file_number, std::string_view filename) {
  maximum_fileno_ =
      maximum_fileno_.has_value()
          ? Fileno(std::max(static_cast<int32_t>(file_number),
                            static_cast<int32_t>(maximum_fileno_.value())))
          : file_number;
  filename_to_fileno_.emplace(std::string(filename), file_number);
  fileno_to_filename_.emplace(file_number, std::string(filename));
}

std::optional<std::string> Package::GetFilename(Fileno file_number) const {
  if (!fileno_to_filename_.contains(file_number)) {
    return std::nullopt;
  }
  return fileno_to_filename_.at(file_number);
}

int64_t Package::GetFunctionNodeCount() const {
  int64_t count = 0;
  for (const auto& f : functions()) {
    count += f->node_count();
  }
  return count;
}

int64_t Package::GetProcNodeCount() const {
  int64_t count = 0;
  for (const auto& f : procs()) {
    count += f->node_count();
  }
  return count;
}

int64_t Package::GetBlockNodeCount() const {
  int64_t count = 0;
  for (const auto& f : blocks()) {
    count += f->node_count();
  }
  return count;
}

int64_t Package::GetNodeCount() const {
  return GetFunctionNodeCount() + GetProcNodeCount() + GetBlockNodeCount();
}

bool Package::IsDefinitelyEqualTo(const Package* other) const {
  auto entry_function_status = GetTopAsFunction();
  if (!entry_function_status.ok()) {
    return false;
  }
  auto other_entry_function_status = other->GetTopAsFunction();
  if (!other_entry_function_status.ok()) {
    return false;
  }
  const Function* entry = entry_function_status.value();
  const Function* other_entry = other_entry_function_status.value();
  return entry->IsDefinitelyEqualTo(other_entry);
}

std::string Package::DumpIr() const {
  std::string out;
  absl::StrAppend(&out, "package ", name(), "\n\n");

  if (!fileno_to_filename_.empty()) {
    std::list<xls::Fileno> filenos;
    for (const auto& [fileno, filename] : fileno_to_filename_) {
      filenos.push_back(fileno);
    }
    filenos.sort();
    // output in sorted order to be deterministic
    for (const auto& fileno : filenos) {
      std::string_view filename = fileno_to_filename_.at(fileno);
      absl::StrAppend(&out, "file_number ", static_cast<int32_t>(fileno), " ",
                      "\"", filename, "\"\n");
    }
    absl::StrAppend(&out, "\n");
  }

  if (!channels().empty()) {
    for (Channel* channel : channels()) {
      absl::StrAppend(&out, channel->ToString(), "\n");
    }
    absl::StrAppend(&out, "\n");
  }
  std::optional<FunctionBase*> top = GetTop();
  auto append_ir_with_attributes = [&top, &out](FunctionBase* fb) {
    std::string_view attribute_prefix;
    std::string_view attribute_suffix;
    std::vector<std::string> attribute_strings = fb->AttributeIrStrings();
    if (!attribute_strings.empty()) {
      attribute_prefix = "#[";
      attribute_suffix = "]\n";
    }
    std::string_view top_prefix;
    if (top.has_value() && top.value() == fb) {
      top_prefix = "top ";
    }

    absl::StrAppend(&out, attribute_prefix,
                    absl::StrJoin(attribute_strings, ", "), attribute_suffix,
                    top_prefix, fb->DumpIr(), "\n");
  };
  // Our parser relies on everything being in post-order. Ensure that here.
  for (FunctionBase* fb : FunctionsInPostOrder(this)) {
    append_ir_with_attributes(fb);
  }
  // We don't include the trailing newline, drop it here.
  CHECK_EQ(out.back(), '\n');
  out.pop_back();
  return out;
}

std::ostream& operator<<(std::ostream& os, const Package& package) {
  os << package.DumpIr();
  return os;
}

std::vector<std::string> Package::GetFunctionNames() const {
  std::vector<std::string> names;
  names.reserve(functions_.size());
  for (const std::unique_ptr<Function>& function : functions_) {
    names.push_back(function->name());
  }
  std::sort(names.begin(), names.end());
  return names;
}

namespace {

absl::Status VerifyValuesAreType(absl::Span<const Value> values, Type* type) {
  for (const Value& value : values) {
    if (!ValueConformsToType(value, type)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Initial value does not match channel type %s: %s",
                          type->ToString(), value.ToString()));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<StreamingChannel*> Package::CreateStreamingChannel(
    std::string_view name, ChannelOps supported_ops, Type* type,
    absl::Span<const Value> initial_values, ChannelConfig channel_config,
    FlowControl flow_control, ChannelStrictness strictness,
    std::optional<int64_t> id) {
  return CreateStreamingChannelInProc(name, supported_ops, type,
                                      /*proc=*/nullptr, initial_values,
                                      channel_config, flow_control, strictness,
                                      id);
}

absl::StatusOr<StreamingChannel*> Package::CreateStreamingChannelInProc(
    std::string_view name, ChannelOps supported_ops, Type* type, Proc* proc,
    absl::Span<const Value> initial_values, ChannelConfig channel_config,
    FlowControl flow_control, ChannelStrictness strictness,
    std::optional<int64_t> id) {
  XLS_RETURN_IF_ERROR(VerifyValuesAreType(initial_values, type));
  int64_t actual_id = id.has_value() ? id.value() : next_channel_id_;
  auto channel = std::make_unique<StreamingChannel>(
      name, actual_id, supported_ops, type, initial_values, channel_config,
      flow_control, strictness);
  StreamingChannel* channel_ptr = channel.get();
  XLS_RETURN_IF_ERROR(AddChannel(std::move(channel), proc));
  return channel_ptr;
}

absl::StatusOr<SingleValueChannel*> Package::CreateSingleValueChannel(
    std::string_view name, ChannelOps supported_ops, Type* type,
    std::optional<int64_t> id) {
  return Package::CreateSingleValueChannelInProc(name, supported_ops, type,
                                                 /*proc=*/nullptr, id);
}

absl::StatusOr<SingleValueChannel*> Package::CreateSingleValueChannelInProc(
    std::string_view name, ChannelOps supported_ops, Type* type, Proc* proc,
    std::optional<int64_t> id) {
  int64_t actual_id = id.has_value() ? id.value() : next_channel_id_;
  auto channel = std::make_unique<SingleValueChannel>(name, actual_id,
                                                      supported_ops, type);
  SingleValueChannel* channel_ptr = channel.get();
  XLS_RETURN_IF_ERROR(AddChannel(std::move(channel), proc));
  return channel_ptr;
}

absl::Status Package::RemoveChannel(Channel* channel) {
  // First check that the channel is owned by this package.
  auto it = std::find(channel_vec_.begin(), channel_vec_.end(), channel);
  XLS_RET_CHECK(it != channel_vec_.end()) << "Channel not owned by package";

  // Check that no send/receive nodes are associated with the channel.
  // TODO(https://github.com/google/xls/issues/411) 2012/04/24 Avoid iterating
  // through all the nodes after channels are mapped to send/receive nodes.
  for (const auto& proc : procs()) {
    if (proc->is_new_style_proc()) {
      continue;
    }
    for (Node* node : proc->nodes()) {
      if ((node->Is<Send>() &&
           node->As<Send>()->channel_name() == channel->name()) ||
          (node->Is<Receive>() &&
           node->As<Receive>()->channel_name() == channel->name())) {
        return absl::InternalError(absl::StrFormat(
            "Channel %s (id=%d) cannot be removed because it "
            "is used by node %v in %v",
            channel->name(), channel->id(), *node, *node->function_base()));
      }
    }
  }

  // Remove from channel vector.
  channel_vec_.erase(it);

  // Remove from channel map.
  XLS_RET_CHECK(channels_.contains(channel->name()));
  channels_.erase(channel->name());

  return absl::OkStatus();
}

absl::Status Package::AddChannel(std::unique_ptr<Channel> channel, Proc* proc) {
  if (proc != nullptr) {
    next_channel_id_ = std::max(next_channel_id_, channel->id() + 1);
    return proc->AddChannel(std::move(channel)).status();
  }
  std::string name{channel->name()};
  auto [channel_it, inserted] = channels_.insert({name, std::move(channel)});
  if (!inserted) {
    return absl::InternalError(
        absl::StrFormat("Channel already exists with name `%s`.", name));
  }
  Channel* channel_ptr = channel_it->second.get();

  // Verify the channel id is unique.
  for (Channel* ch : channel_vec_) {
    if (ch->id() == channel_ptr->id()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel already exists with id %d", ch->id()));
    }
  }

  // The channel name and all data element names must be valid identifiers.
  if (!NameUniquer::IsValidIdentifier(channel_ptr->name())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid channel name: \"%s\"", channel_ptr->name()));
  }

  // Add pointer to newly added channel to the channel vector and resort it by
  // ID.
  channel_vec_.push_back(channel_ptr);
  std::sort(channel_vec_.begin(), channel_vec_.end(),
            [](Channel* a, Channel* b) { return a->id() < b->id(); });

  next_channel_id_ = std::max(next_channel_id_, channel_ptr->id() + 1);
  return absl::OkStatus();
}

absl::StatusOr<Channel*> Package::GetChannel(int64_t id) const {
  XLS_RET_CHECK(!ChannelsAreProcScoped());
  for (Channel* ch : channels()) {
    if (ch->id() == id) {
      return ch;
    }
  }
  return absl::NotFoundError(absl::StrFormat("No channel with id %d", id));
}

std::vector<std::string> Package::GetChannelNames() const {
  CHECK(!ChannelsAreProcScoped());
  std::vector<std::string> names;
  names.reserve(channels().size());
  for (Channel* ch : channels()) {
    names.push_back(std::string{ch->name()});
  }
  return names;
}

absl::StatusOr<Channel*> Package::GetChannel(std::string_view name) const {
  XLS_RET_CHECK(!ChannelsAreProcScoped());
  auto it = channels_.find(name);
  if (it != channels_.end()) {
    return it->second.get();
  }
  return absl::NotFoundError(
      absl::StrFormat("No channel with name `%s`", name));
}

bool Package::ChannelsAreProcScoped() const {
  if (!channels_.empty()) {
    return false;
  }
  if (procs_.empty()) {
    return false;
  }
  // The verifier checks that all procs are the same style so just check the
  // first proc.
  return procs_.front()->is_new_style_proc();
}

absl::StatusOr<Channel*> Package::CloneChannel(
    Channel* channel, std::string_view name,
    const CloneChannelOverrides& overrides) {
  XLS_ASSIGN_OR_RETURN(auto* new_channel_type,
                       this->MapTypeFromOtherPackage(channel->type()));
  switch (channel->kind()) {
    case ChannelKind::kSingleValue: {
      if (overrides.initial_values().has_value() ||
          overrides.channel_config().has_value() ||
          overrides.flow_control().has_value()) {
        return absl::InvalidArgumentError(
            "Cannot clone single value channel with streaming channel "
            "parameter overrides.");
      }
      XLS_ASSIGN_OR_RETURN(
          auto new_channel,
          this->CreateSingleValueChannel(
              name,
              overrides.supported_ops().value_or(channel->supported_ops()),
              new_channel_type));
      return new_channel;
    }
    case ChannelKind::kStreaming: {
      auto streaming_channel = dynamic_cast<StreamingChannel*>(channel);
      if (streaming_channel == nullptr) {
        return absl::InternalError(
            absl::StrFormat("Channel %s had kind kStreaming, but could not "
                            "be cast to StreamingChannel",
                            channel->name()));
      }
      ChannelConfig channel_config;
      if (overrides.channel_config().has_value()) {
        channel_config = *overrides.channel_config();
      } else {
        channel_config = streaming_channel->channel_config();
      }
      XLS_ASSIGN_OR_RETURN(
          auto new_channel,
          this->CreateStreamingChannel(
              name,
              overrides.supported_ops().value_or(channel->supported_ops()),
              new_channel_type,
              overrides.initial_values().value_or(channel->initial_values()),
              channel_config,
              overrides.flow_control().value_or(
                  streaming_channel->GetFlowControl()),
              overrides.strictness().value_or(
                  streaming_channel->GetStrictness())));
      return new_channel;
    }
  }
}

absl::StatusOr<FunctionBase*> FindTop(Package* p,
                                      std::optional<std::string_view> top_str) {
  if (top_str.has_value() && !top_str->empty()) {
    XLS_RETURN_IF_ERROR(p->SetTopByName(top_str.value()));
  }

  // Default to the top entity if nothing is specified.
  std::optional<FunctionBase*> top = p->GetTop();
  if (!top.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Top entity not set for package: %s.", p->name()));
  }
  return top.value();
}

/* static */ TransformMetrics TransformMetrics::FromProto(
    const TransformMetricsProto& proto) {
  TransformMetrics ret;
  ret.nodes_added = proto.nodes_added();
  ret.nodes_removed = proto.nodes_removed();
  ret.nodes_replaced = proto.nodes_replaced();
  ret.operands_replaced = proto.operands_replaced();
  ret.operands_removed = proto.operands_removed();
  return ret;
}

TransformMetrics TransformMetrics::operator+(
    const TransformMetrics& other) const {
  return TransformMetrics{
      .nodes_added = nodes_added + other.nodes_added,
      .nodes_removed = nodes_removed + other.nodes_removed,
      .nodes_replaced = nodes_replaced + other.nodes_replaced,
      .operands_replaced = operands_replaced + other.operands_replaced,
      .operands_removed = operands_removed + other.operands_removed,
  };
}

TransformMetrics TransformMetrics::operator-(
    const TransformMetrics& other) const {
  return TransformMetrics{
      .nodes_added = nodes_added - other.nodes_added,
      .nodes_removed = nodes_removed - other.nodes_removed,
      .nodes_replaced = nodes_replaced - other.nodes_replaced,
      .operands_replaced = operands_replaced - other.operands_replaced,
      .operands_removed = operands_removed - other.operands_removed,
  };
}

std::string TransformMetrics::ToString() const {
  return absl::StrFormat(
      "{ nodes added: %d, nodes removed: %d, nodes replaced: %d, operands "
      "replaced: %d, operands removed: %d }",
      nodes_added, nodes_removed, nodes_replaced, operands_replaced,
      operands_removed);
}

TransformMetricsProto TransformMetrics::ToProto() const {
  TransformMetricsProto ret;
  ret.set_nodes_added(nodes_added);
  ret.set_nodes_removed(nodes_removed);
  ret.set_nodes_replaced(nodes_replaced);
  ret.set_operands_replaced(operands_replaced);
  ret.set_operands_removed(operands_removed);
  return ret;
}

}  // namespace xls
