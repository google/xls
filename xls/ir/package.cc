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

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strong_int.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"

namespace xls {

Package::Package(std::string_view name) : name_(name) {
  owned_types_.insert(&token_type_);
}

Package::~Package() {}

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
    std::string_view name) {
  std::vector<FunctionBase*> fbs = GetFunctionBases();
  int64_t count = std::count_if(
      fbs.begin(), fbs.end(),
      [name](const FunctionBase* fb) { return fb->name() == name; });
  if (count == 0) {
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
  if (count == 1) {
    auto fb_iter = std::find_if(
        fbs.begin(), fbs.end(),
        [name](const FunctionBase* fb) { return fb->name() == name; });
    return *fb_iter;
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

std::string Package::SourceLocationToString(const SourceLocation loc) {
  const std::string unknown = "UNKNOWN";
  std::string_view filename =
      fileno_to_filename_.find(loc.fileno()) != fileno_to_filename_.end()
          ? fileno_to_filename_.at(loc.fileno())
          : unknown;
  return absl::StrFormat("%s:%d", filename, loc.lineno().value());
}

absl::StatusOr<Type*> Package::MapTypeFromOtherPackage(
    Type* other_package_type) {
  // Package already owns this type.
  if (IsOwnedType(other_package_type)) {
    return other_package_type;
  }

  if (other_package_type->IsBits()) {
    const BitsType* bits = other_package_type->AsBitsOrDie();
    return GetBitsType(bits->bit_count());

  } else if (other_package_type->IsArray()) {
    const ArrayType* array = other_package_type->AsArrayOrDie();
    XLS_ASSIGN_OR_RETURN(Type * elem_type,
                         MapTypeFromOtherPackage(array->element_type()));
    return GetArrayType(array->size(), elem_type);

  } else if (other_package_type->IsTuple()) {
    const TupleType* tuple = other_package_type->AsTupleOrDie();
    std::vector<Type*> member_types;
    member_types.reserve(tuple->size());
    for (auto* elem_type : tuple->element_types()) {
      XLS_ASSIGN_OR_RETURN(Type * new_elem_type,
                           MapTypeFromOtherPackage(elem_type));
      member_types.push_back(new_elem_type);
    }
    return GetTupleType(member_types);

  } else if (other_package_type->IsToken()) {
    return GetTokenType();

  } else {
    return absl::InternalError("Unsupported type.");
  }
}

BitsType* Package::GetBitsType(int64_t bit_count) {
  if (bit_count_to_type_.find(bit_count) != bit_count_to_type_.end()) {
    return &bit_count_to_type_.at(bit_count);
  }
  auto it = bit_count_to_type_.emplace(bit_count, BitsType(bit_count));
  BitsType* new_type = &(it.first->second);
  owned_types_.insert(new_type);
  return new_type;
}

ArrayType* Package::GetArrayType(int64_t size, Type* element_type) {
  ArrayKey key{size, element_type};
  if (array_types_.find(key) != array_types_.end()) {
    return &array_types_.at(key);
  }
  XLS_CHECK(IsOwnedType(element_type))
      << "Type is not owned by package: " << *element_type;
  auto it = array_types_.emplace(key, ArrayType(size, element_type));
  ArrayType* new_type = &(it.first->second);
  owned_types_.insert(new_type);
  return new_type;
}

TupleType* Package::GetTupleType(absl::Span<Type* const> element_types) {
  TypeVec key(element_types.begin(), element_types.end());
  if (tuple_types_.find(key) != tuple_types_.end()) {
    return &tuple_types_.at(key);
  }
  for (const Type* element_type : element_types) {
    XLS_CHECK(IsOwnedType(element_type))
        << "Type is not owned by package: " << *element_type;
  }
  auto it = tuple_types_.emplace(key, TupleType(element_types));
  TupleType* new_type = &(it.first->second);
  owned_types_.insert(new_type);
  return new_type;
}

TokenType* Package::GetTokenType() { return &token_type_; }

FunctionType* Package::GetFunctionType(absl::Span<Type* const> args_types,
                                       Type* return_type) {
  std::string key = FunctionType(args_types, return_type).ToString();
  if (function_types_.find(key) != function_types_.end()) {
    return &function_types_.at(key);
  }
  for (Type* t : args_types) {
    XLS_CHECK(IsOwnedType(t))
        << "Parameter type is not owned by package: " << t->ToString();
  }
  auto it = function_types_.emplace(key, FunctionType(args_types, return_type));
  FunctionType* new_type = &(it.first->second);
  owned_function_types_.insert(new_type);
  return new_type;
}

absl::StatusOr<Type*> Package::GetTypeFromProto(const TypeProto& proto) {
  if (!proto.has_type_enum()) {
    return absl::InvalidArgumentError("Missing type_enum field in TypeProto.");
  }
  if (proto.type_enum() == TypeProto::BITS) {
    if (!proto.has_bit_count() || proto.bit_count() < 0) {
      return absl::InvalidArgumentError(
          "Missing or invalid bit_count field in TypeProto.");
    }
    return GetBitsType(proto.bit_count());
  }
  if (proto.type_enum() == TypeProto::TUPLE) {
    std::vector<Type*> elements;
    for (const TypeProto& element_proto : proto.tuple_elements()) {
      XLS_ASSIGN_OR_RETURN(Type * element, GetTypeFromProto(element_proto));
      elements.push_back(element);
    }
    return GetTupleType(elements);
  }
  if (proto.type_enum() == TypeProto::ARRAY) {
    if (!proto.has_array_size() || proto.array_size() < 0) {
      return absl::InvalidArgumentError(
          "Missing or invalid array_size field in TypeProto.");
    }
    if (!proto.has_array_element()) {
      return absl::InvalidArgumentError(
          "Missing array_element field in TypeProto.");
    }
    XLS_ASSIGN_OR_RETURN(Type * element_type,
                         GetTypeFromProto(proto.array_element()));
    return GetArrayType(proto.array_size(), element_type);
  }
  if (proto.type_enum() == TypeProto::TOKEN) {
    return GetTokenType();
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid type_enum value in TypeProto: %d", proto.type_enum()));
}

absl::StatusOr<FunctionType*> Package::GetFunctionTypeFromProto(
    const FunctionTypeProto& proto) {
  std::vector<Type*> param_types;
  for (const TypeProto& param_proto : proto.parameters()) {
    XLS_ASSIGN_OR_RETURN(Type * param_type, GetTypeFromProto(param_proto));
    param_types.push_back(param_type);
  }
  if (!proto.has_return_type()) {
    return absl::InvalidArgumentError(
        "Missing return_type field in FunctionTypeProto.");
  }
  XLS_ASSIGN_OR_RETURN(Type * return_type,
                       GetTypeFromProto(proto.return_type()));
  return GetFunctionType(param_types, return_type);
}

Type* Package::GetTypeForValue(const Value& value) {
  switch (value.kind()) {
    case ValueKind::kBits:
      return GetBitsType(value.bits().bit_count());
    case ValueKind::kTuple: {
      std::vector<Type*> element_types;
      for (const Value& value : value.elements()) {
        element_types.push_back(GetTypeForValue(value));
      }
      return GetTupleType(element_types);
    }
    case ValueKind::kArray: {
      // No element type can be inferred for 0-element arrays.
      if (value.empty()) {
        return GetArrayType(0, nullptr);
      }
      return GetArrayType(value.size(), GetTypeForValue(value.elements()[0]));
    }
    case ValueKind::kToken:
      return GetTokenType();
    case ValueKind::kInvalid:
      break;
  }
  XLS_LOG(FATAL) << "Invalid value for type extraction.";
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

int64_t Package::GetNodeCount() const {
  int64_t count = 0;
  for (const auto& f : functions()) {
    count += f->node_count();
  }
  return count;
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
    for (const auto& fileno  : filenos) {
      std::string filename = fileno_to_filename_.at(fileno);
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
  std::vector<std::string> function_dumps;
  std::optional<FunctionBase*> top = GetTop();
  for (auto& function : functions()) {
    std::string prefix = "";
    if (top.has_value() && top.value() == function.get()) {
      prefix = "top ";
    }
    function_dumps.push_back(absl::StrCat(prefix, function->DumpIr()));
  }
  for (auto& proc : procs()) {
    std::string prefix = "";
    if (top.has_value() && top.value() == proc.get()) {
      prefix = "top ";
    }
    function_dumps.push_back(absl::StrCat(prefix, proc->DumpIr()));
  }
  for (auto& block : blocks()) {
    std::string prefix = "";
    if (top.has_value() && top.value() == block.get()) {
      prefix = "top ";
    }
    function_dumps.push_back(absl::StrCat(prefix, block->DumpIr()));
  }
  absl::StrAppend(&out, absl::StrJoin(function_dumps, "\n"));
  return out;
}

std::ostream& operator<<(std::ostream& os, const Package& package) {
  os << package.DumpIr();
  return os;
}

absl::flat_hash_map<std::string, Function*> Package::GetFunctionByName() {
  absl::flat_hash_map<std::string, Function*> name_to_function;
  for (std::unique_ptr<Function>& function : functions_) {
    name_to_function[function->name()] = function.get();
  }
  return name_to_function;
}

std::vector<std::string> Package::GetFunctionNames() const {
  std::vector<std::string> names;
  for (const std::unique_ptr<Function>& function : functions_) {
    names.push_back(function->name());
  }
  std::sort(names.begin(), names.end());
  return names;
}

bool Package::HasFunctionWithName(std::string_view target) const {
  for (const std::unique_ptr<Function>& function : functions_) {
    if (function->name() == target) {
      return true;
    }
  }
  return false;
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
    absl::Span<const Value> initial_values, std::optional<int64_t> fifo_depth,
    FlowControl flow_control, const ChannelMetadataProto& metadata,
    std::optional<int64_t> id) {
  XLS_RETURN_IF_ERROR(VerifyValuesAreType(initial_values, type));
  int64_t actual_id = id.has_value() ? id.value() : next_channel_id_;
  auto channel = std::make_unique<StreamingChannel>(
      name, actual_id, supported_ops, type, initial_values, fifo_depth,
      flow_control, metadata);
  StreamingChannel* channel_ptr = channel.get();
  XLS_RETURN_IF_ERROR(AddChannel(std::move(channel)));
  return channel_ptr;
}

absl::StatusOr<SingleValueChannel*> Package::CreateSingleValueChannel(
    std::string_view name, ChannelOps supported_ops, Type* type,
    const ChannelMetadataProto& metadata, std::optional<int64_t> id) {
  int64_t actual_id = id.has_value() ? id.value() : next_channel_id_;
  auto channel = std::make_unique<SingleValueChannel>(
      name, actual_id, supported_ops, type, metadata);
  SingleValueChannel* channel_ptr = channel.get();
  XLS_RETURN_IF_ERROR(AddChannel(std::move(channel)));
  return channel_ptr;
}

absl::Status Package::RemoveChannel(Channel* channel) {
  // First check that the channel is owned by this package.
  auto it = std::find(channel_vec_.begin(), channel_vec_.end(), channel);
  XLS_RET_CHECK(it != channel_vec_.end()) << "Channel not owned by package";

  // Check that no send/receive nodes are associted with the channel.
  // TODO(https://github.com/google/xls/issues/411) 2012/04/24 Avoid iterating
  // through all the nodes after channels are mapped to send/receive nodes.
  for (const auto& proc : procs()) {
    for (Node* node : proc->nodes()) {
      if ((node->Is<Send>() &&
           node->As<Send>()->channel_id() == channel->id()) ||
          (node->Is<Receive>() &&
           node->As<Receive>()->channel_id() == channel->id())) {
        return absl::InternalError(absl::StrFormat(
            "Channel %s (%d) cannot be removed because it is used by node %s",
            channel->name(), channel->id(), node->GetName()));
      }
    }
  }

  // Remove from channel vector.
  channel_vec_.erase(it);

  // Remove from channel map.
  XLS_RET_CHECK(channels_.contains(channel->id()));
  channels_.erase(channel->id());

  return absl::OkStatus();
}

absl::Status Package::AddChannel(std::unique_ptr<Channel> channel) {
  int64_t id = channel->id();
  auto [channel_it, inserted] = channels_.insert({id, std::move(channel)});
  if (!inserted) {
    return absl::InternalError(
        absl::StrFormat("Channel already exists with id %d.", id));
  }
  Channel* channel_ptr = channel_it->second.get();

  // Verify the channel name is unique.
  for (Channel* ch : channel_vec_) {
    if (ch->name() == channel_ptr->name()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel already exists with name \"%s\"", ch->name()));
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

  next_channel_id_ = std::max(next_channel_id_, id + 1);
  return absl::OkStatus();
}

absl::StatusOr<Channel*> Package::GetChannel(int64_t id) const {
  if (channels_.find(id) == channels_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No channel with id %d (package has %d channels).", id,
                        channels_.size()));
  }
  return channels_.at(id).get();
}

absl::StatusOr<Channel*> Package::GetChannel(std::string_view name) const {
  for (Channel* ch : channels()) {
    if (ch->name() == name) {
      return ch;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("No channel with name '%s' (package has %d channels).",
                      name, channels().size()));
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

}  // namespace xls
