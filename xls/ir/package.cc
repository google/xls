// Copyright 2020 Google LLC
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

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strong_int.h"
#include "xls/ir/function.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {
constexpr char kMain[] = "main";
}

Package::Package(absl::string_view name,
                 absl::optional<absl::string_view> entry)
    : entry_(entry), name_(name) {
  owned_types_.insert(&token_type_);
}

Package::~Package() {}

Function* Package::AddFunction(std::unique_ptr<Function> f) {
  functions_.push_back(std::move(f));
  return functions_.back().get();
}

xabsl::StatusOr<Function*> Package::GetFunction(
    absl::string_view func_name) const {
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

void Package::DeleteDeadFunctions(absl::Span<Function* const> dead_funcs) {
  std::vector<std::unique_ptr<Function>> to_unlink;
  for (std::unique_ptr<Function>& f : functions_) {
    if (std::find(dead_funcs.begin(), dead_funcs.end(), f.get()) !=
        dead_funcs.end()) {
      XLS_VLOG(1) << "Function is dead: " << f->name();
      to_unlink.push_back(std::move(f));
      f = nullptr;
    }
  }

  // Destruct all the functions here instead of when they go out of scope,
  // just in  case there's meaningful logging in the destructor.
  to_unlink.clear();

  // Get rid of nullptrs we made in the functions vector.
  functions_.erase(std::remove(functions_.begin(), functions_.end(), nullptr),
                   functions_.end());
}

xabsl::StatusOr<Function*> Package::EntryFunction() {
  auto by_name = GetFunctionByName();

  if (entry_.has_value()) {
    auto it = by_name.find(entry_.value());
    if (it != by_name.end()) {
      return it->second;
    }
    std::string available =
        absl::StrJoin(by_name.begin(), by_name.end(), ", ",
                      [](std::string* out,
                         const std::pair<std::string, const Function*>& item) {
                        absl::StrAppend(out, "\"", item.first, "\"");
                      });

    return absl::NotFoundError(
        absl::StrFormat("Could not find entry function for this package; "
                        "tried: [\"%s\"]; available: %s",
                        entry_.value(), available));
  }

  // Try a few possibilities of names for the canonical entry function.
  const std::vector<std::string> to_try = {
      kMain,
      name(),
      absl::StrCat("__", name(), "__", kMain),
      absl::StrCat("__", name(), "__", name()),
  };

  for (const std::string& attempt : to_try) {
    auto it = by_name.find(attempt);
    if (it != by_name.end()) {
      return it->second;
    }
  }

  // Finally we use the only function if only one exists.
  if (functions_.size() == 1) {
    return functions_.front().get();
  }
  auto quote = [](std::string* out, const std::string& s) {
    absl::StrAppend(out, "\"", s, "\"");
  };
  return absl::NotFoundError(absl::StrFormat(
      "Could not find an entry function for the \"%s\" package; "
      "attempted: [%s]",
      name(), absl::StrJoin(to_try, ", ", quote)));
}

xabsl::StatusOr<const Function*> Package::EntryFunction() const {
  XLS_ASSIGN_OR_RETURN(Function * f,
                       const_cast<Package*>(this)->EntryFunction());
  return f;
}

SourceLocation Package::AddSourceLocation(absl::string_view filename,
                                          Lineno lineno, Colno colno) {
  Fileno this_fileno = GetOrCreateFileno(filename);
  return SourceLocation(this_fileno, lineno, colno);
}

std::string Package::SourceLocationToString(const SourceLocation loc) {
  const std::string unknown = "UNKNOWN";
  absl::string_view filename =
      fileno_to_filename_.find(loc.fileno()) != fileno_to_filename_.end()
          ? fileno_to_filename_.at(loc.fileno())
          : unknown;
  return absl::StrFormat("%s:%d", filename, loc.lineno().value());
}

BitsType* Package::GetBitsType(int64 bit_count) {
  if (bit_count_to_type_.find(bit_count) != bit_count_to_type_.end()) {
    return &bit_count_to_type_.at(bit_count);
  }
  auto it = bit_count_to_type_.emplace(bit_count, BitsType(bit_count));
  BitsType* new_type = &(it.first->second);
  owned_types_.insert(new_type);
  return new_type;
}

ArrayType* Package::GetArrayType(int64 size, Type* element_type) {
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

xabsl::StatusOr<Type*> Package::GetTypeFromProto(const TypeProto& proto) {
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
  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid type_enum value in TypeProto: %d", proto.type_enum()));
}

xabsl::StatusOr<FunctionType*> Package::GetFunctionTypeFromProto(
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

Fileno Package::GetOrCreateFileno(absl::string_view filename) {
  // Attempt to add a new fileno/filename pair to the map.
  auto this_fileno = Fileno(filename_to_fileno_.size());
  if (auto it = filename_to_fileno_.find(std::string(filename));
      it != filename_to_fileno_.end()) {
    return it->second;
  }
  filename_to_fileno_.emplace(std::string(filename), this_fileno);
  fileno_to_filename_.emplace(this_fileno, std::string(filename));

  return this_fileno;
}

int64 Package::GetNodeCount() const {
  int64 count = 0;
  for (const auto& f : functions()) {
    count += f->node_count();
  }
  return count;
}

bool Package::IsDefinitelyEqualTo(const Package* other) const {
  auto entry_function_status = EntryFunction();
  if (!entry_function_status.ok()) {
    return false;
  }
  auto other_entry_function_status = other->EntryFunction();
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
  std::vector<std::string> function_dumps;
  for (auto& function : functions()) {
    function_dumps.push_back(function->DumpIr());
  }
  absl::StrAppend(&out, absl::StrJoin(function_dumps, "\n"));
  return out;
}

std::ostream& operator<<(std::ostream& os, const Package& package) {
  os << package.DumpIr();
  return os;
}

#include "xls/ir/container_hack.inc"

UnorderedMap<std::string, Function*> Package::GetFunctionByName() {
  UnorderedMap<std::string, Function*> name_to_function;
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

}  // namespace xls
