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

#include "xls/public/c_api_dslx.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/visitor.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_from_string.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/replace_invocations.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/public/c_api_impl_helpers.h"

namespace {

struct CallGraphHolder {
  xls::dslx::TypeInfo* type_info;
  std::vector<const xls::dslx::Function*> functions;
  absl::flat_hash_map<const xls::dslx::Function*,
                      std::vector<const xls::dslx::Function*>>
      graph;
};

const struct xls_dslx_type* GetMetaTypeHelper(
    struct xls_dslx_type_info* type_info, xls::dslx::AstNode* cpp_node) {
  CHECK(cpp_node != nullptr);
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  std::optional<xls::dslx::Type*> maybe_type = cpp_type_info->GetItem(cpp_node);
  if (!maybe_type.has_value()) {
    return nullptr;
  }
  CHECK(maybe_type.value() != nullptr);
  // Should always have a metatype as its associated type.
  absl::StatusOr<const xls::dslx::Type*> unwrapped =
      xls::dslx::UnwrapMetaType(*maybe_type.value());
  CHECK_OK(unwrapped);
  return reinterpret_cast<const struct xls_dslx_type*>(*unwrapped);
}

struct InvocationCalleeDataArray {
  InvocationCalleeDataArray() = default;

  explicit InvocationCalleeDataArray(
      std::vector<xls::dslx::InvocationCalleeData> entries_in)
      : entries(std::move(entries_in)) {}

  std::vector<xls::dslx::InvocationCalleeData> entries;
};

template <typename T>
xls::dslx::ModuleMember* FindModuleMemberForNode(T* node) {
  if (node == nullptr) {
    return nullptr;
  }
  xls::dslx::Module* module = node->owner();
  if (module == nullptr) {
    return nullptr;
  }
  for (xls::dslx::ModuleMember& member : module->top()) {
    if (std::holds_alternative<T*>(member) && std::get<T*>(member) == node) {
      return &member;
    }
  }
  return nullptr;
}

}  // namespace

extern "C" {

bool xls_dslx_parametric_env_create(
    const struct xls_dslx_parametric_env_item* items, size_t items_count,
    char** error_out, struct xls_dslx_parametric_env** env_out) {
  CHECK(error_out != nullptr);
  CHECK(env_out != nullptr);
  *error_out = nullptr;
  if (items_count == 0) {
    *env_out = reinterpret_cast<xls_dslx_parametric_env*>(
        new xls::dslx::ParametricEnv());
    return true;
  }

  std::vector<std::pair<std::string, xls::dslx::InterpValue>> v;
  v.reserve(items_count);
  for (size_t i = 0; i < items_count; ++i) {
    const xls_dslx_parametric_env_item& it = items[i];
    CHECK(it.identifier != nullptr);
    CHECK(it.value != nullptr);
    const xls::dslx::InterpValue* iv =
        reinterpret_cast<const xls::dslx::InterpValue*>(it.value);
    v.emplace_back(it.identifier, *iv);
  }

  *env_out = reinterpret_cast<xls_dslx_parametric_env*>(
      new xls::dslx::ParametricEnv(absl::MakeSpan(v)));
  return true;
}

struct xls_dslx_parametric_env* xls_dslx_parametric_env_clone(
    const struct xls_dslx_parametric_env* env) {
  CHECK(env != nullptr);
  const auto* cpp_env = reinterpret_cast<const xls::dslx::ParametricEnv*>(env);
  auto* heap = new xls::dslx::ParametricEnv(*cpp_env);
  return reinterpret_cast<xls_dslx_parametric_env*>(heap);
}

bool xls_dslx_parametric_env_equals(const struct xls_dslx_parametric_env* lhs,
                                    const struct xls_dslx_parametric_env* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::dslx::ParametricEnv*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::dslx::ParametricEnv*>(rhs);
  return *cpp_lhs == *cpp_rhs;
}

bool xls_dslx_parametric_env_less_than(
    const struct xls_dslx_parametric_env* lhs,
    const struct xls_dslx_parametric_env* rhs) {
  CHECK(lhs != nullptr);
  CHECK(rhs != nullptr);
  const auto* cpp_lhs = reinterpret_cast<const xls::dslx::ParametricEnv*>(lhs);
  const auto* cpp_rhs = reinterpret_cast<const xls::dslx::ParametricEnv*>(rhs);
  const auto& lhs_bindings = cpp_lhs->bindings();
  const auto& rhs_bindings = cpp_rhs->bindings();
  const int64_t common = std::min(lhs_bindings.size(), rhs_bindings.size());
  for (int64_t i = 0; i < common; ++i) {
    const auto& lhs_item = lhs_bindings[i];
    const auto& rhs_item = rhs_bindings[i];
    if (lhs_item.identifier < rhs_item.identifier) {
      return true;
    }
    if (rhs_item.identifier < lhs_item.identifier) {
      return false;
    }
    if (lhs_item.value < rhs_item.value) {
      return true;
    }
    if (rhs_item.value < lhs_item.value) {
      return false;
    }
  }
  return lhs_bindings.size() < rhs_bindings.size();
}

uint64_t xls_dslx_parametric_env_hash(
    const struct xls_dslx_parametric_env* env) {
  CHECK(env != nullptr);
  const auto* cpp_env = reinterpret_cast<const xls::dslx::ParametricEnv*>(env);
  return static_cast<uint64_t>(absl::HashOf(*cpp_env));
}

char* xls_dslx_parametric_env_to_string(
    const struct xls_dslx_parametric_env* env) {
  CHECK(env != nullptr);
  const auto* cpp_env = reinterpret_cast<const xls::dslx::ParametricEnv*>(env);
  return xls::ToOwnedCString(cpp_env->ToString());
}

void xls_dslx_parametric_env_free(struct xls_dslx_parametric_env* env) {
  delete reinterpret_cast<xls::dslx::ParametricEnv*>(env);
}

int64_t xls_dslx_parametric_env_get_binding_count(
    const struct xls_dslx_parametric_env* env) {
  CHECK(env != nullptr);
  auto* cpp_env = reinterpret_cast<const xls::dslx::ParametricEnv*>(env);
  return cpp_env->size();
}

const char* xls_dslx_parametric_env_get_binding_identifier(
    const struct xls_dslx_parametric_env* env, int64_t index) {
  CHECK(env != nullptr);
  auto* cpp_env = reinterpret_cast<const xls::dslx::ParametricEnv*>(env);
  const xls::dslx::ParametricEnvItem& item = cpp_env->bindings().at(index);
  return item.identifier.c_str();
}

struct xls_dslx_interp_value* xls_dslx_parametric_env_get_binding_value(
    const struct xls_dslx_parametric_env* env, int64_t index) {
  CHECK(env != nullptr);
  auto* cpp_env = reinterpret_cast<const xls::dslx::ParametricEnv*>(env);
  const xls::dslx::ParametricEnvItem& item = cpp_env->bindings().at(index);
  return reinterpret_cast<xls_dslx_interp_value*>(
      const_cast<xls::dslx::InterpValue*>(&item.value));
}

// InterpValue simple constructors
struct xls_dslx_interp_value* xls_dslx_interp_value_make_ubits(
    int64_t bit_count, uint64_t value) {
  auto* iv = new xls::dslx::InterpValue(xls::dslx::InterpValue::MakeUBits(
      bit_count, static_cast<int64_t>(value)));
  return reinterpret_cast<xls_dslx_interp_value*>(iv);
}

struct xls_dslx_interp_value* xls_dslx_interp_value_make_sbits(
    int64_t bit_count, int64_t value) {
  auto* iv = new xls::dslx::InterpValue(
      xls::dslx::InterpValue::MakeSBits(bit_count, value));
  return reinterpret_cast<xls_dslx_interp_value*>(iv);
}

bool xls_dslx_interp_value_make_enum(
    struct xls_dslx_enum_def* def, bool is_signed, const struct xls_bits* bits,
    char** error_out, struct xls_dslx_interp_value** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  *error_out = nullptr;
  auto* enum_def = reinterpret_cast<xls::dslx::EnumDef*>(def);
  const xls::Bits* cpp_bits = reinterpret_cast<const xls::Bits*>(bits);
  auto iv = xls::dslx::InterpValue::MakeEnum(*cpp_bits, is_signed, enum_def);
  *result_out = reinterpret_cast<xls_dslx_interp_value*>(
      new xls::dslx::InterpValue(std::move(iv)));
  return true;
}

bool xls_dslx_interp_value_from_string(
    const char* text, const char* dslx_stdlib_path, char** error_out,
    struct xls_dslx_interp_value** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  *error_out = nullptr;
  auto status_or = xls::dslx::InterpValueFromString(
      std::string_view{text}, std::filesystem::path{dslx_stdlib_path});
  if (!status_or.ok()) {
    *result_out = nullptr;
    *error_out = xls::ToOwnedCString(status_or.status().ToString());
    return false;
  }
  *result_out = reinterpret_cast<xls_dslx_interp_value*>(
      new xls::dslx::InterpValue(std::move(status_or.value())));
  return true;
}

bool xls_dslx_interp_value_make_tuple(
    size_t element_count, struct xls_dslx_interp_value** elements,
    char** error_out, struct xls_dslx_interp_value** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  *error_out = nullptr;
  std::vector<xls::dslx::InterpValue> vec;
  vec.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    CHECK(elements[i] != nullptr);
    auto* iv = reinterpret_cast<xls::dslx::InterpValue*>(elements[i]);
    vec.push_back(*iv);
  }
  auto value = xls::dslx::InterpValue::MakeTuple(std::move(vec));
  *result_out = reinterpret_cast<xls_dslx_interp_value*>(
      new xls::dslx::InterpValue(std::move(value)));
  return true;
}

bool xls_dslx_interp_value_make_array(
    size_t element_count, struct xls_dslx_interp_value** elements,
    char** error_out, struct xls_dslx_interp_value** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  *error_out = nullptr;
  std::vector<xls::dslx::InterpValue> vec;
  vec.reserve(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    CHECK(elements[i] != nullptr);
    auto* iv = reinterpret_cast<xls::dslx::InterpValue*>(elements[i]);
    vec.push_back(*iv);
  }
  absl::StatusOr<xls::dslx::InterpValue> arr =
      xls::dslx::InterpValue::MakeArray(std::move(vec));
  if (!arr.ok()) {
    *result_out = nullptr;
    *error_out = xls::ToOwnedCString(arr.status().ToString());
    return false;
  }
  *result_out = reinterpret_cast<xls_dslx_interp_value*>(
      new xls::dslx::InterpValue(std::move(*arr)));
  return true;
}

struct xls_dslx_interp_value* xls_dslx_interp_value_clone(
    const struct xls_dslx_interp_value* value) {
  CHECK(value != nullptr);
  const auto* cpp_interp_value =
      reinterpret_cast<const xls::dslx::InterpValue*>(value);
  auto* heap = new xls::dslx::InterpValue(*cpp_interp_value);
  return reinterpret_cast<xls_dslx_interp_value*>(heap);
}

struct xls_dslx_import_data* xls_dslx_import_data_create(
    const char* dslx_stdlib_path, const char* additional_search_paths[],
    size_t additional_search_paths_count) {
  std::filesystem::path cpp_stdlib_path{dslx_stdlib_path};
  std::vector<std::filesystem::path> cpp_additional_search_paths =
      xls::ToCppPaths(additional_search_paths, additional_search_paths_count);
  xls::dslx::ImportData import_data = CreateImportData(
      cpp_stdlib_path, cpp_additional_search_paths, xls::dslx::kAllWarningsSet,
      std::make_unique<xls::dslx::RealFilesystem>());
  return reinterpret_cast<xls_dslx_import_data*>(
      new xls::dslx::ImportData{std::move(import_data)});
}

void xls_dslx_import_data_free(struct xls_dslx_import_data* x) {
  delete reinterpret_cast<xls::dslx::ImportData*>(x);
}

void xls_dslx_typechecked_module_free(struct xls_dslx_typechecked_module* tm) {
  delete reinterpret_cast<xls::dslx::TypecheckedModule*>(tm);
}

struct xls_dslx_module* xls_dslx_typechecked_module_get_module(
    struct xls_dslx_typechecked_module* tm) {
  auto* cpp_tm = reinterpret_cast<xls::dslx::TypecheckedModule*>(tm);
  xls::dslx::Module* cpp_module = cpp_tm->module;
  return reinterpret_cast<xls_dslx_module*>(cpp_module);
}

struct xls_dslx_type_info* xls_dslx_typechecked_module_get_type_info(
    struct xls_dslx_typechecked_module* tm) {
  auto* cpp_tm = reinterpret_cast<xls::dslx::TypecheckedModule*>(tm);
  xls::dslx::TypeInfo* cpp_type_info = cpp_tm->type_info;
  return reinterpret_cast<xls_dslx_type_info*>(cpp_type_info);
}

struct xls_dslx_type_info* xls_dslx_type_info_get_imported_type_info(
    struct xls_dslx_type_info* type_info, struct xls_dslx_module* module) {
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  std::optional<xls::dslx::TypeInfo*> imported =
      cpp_type_info->GetImportedTypeInfo(cpp_module);
  if (!imported.has_value()) {
    return nullptr;
  }
  return reinterpret_cast<xls_dslx_type_info*>(*imported);
}

bool xls_dslx_parse_and_typecheck(
    const char* text, const char* path, const char* module_name,
    struct xls_dslx_import_data* import_data, char** error_out,
    struct xls_dslx_typechecked_module** result_out) {
  auto* cpp_import_data = reinterpret_cast<xls::dslx::ImportData*>(import_data);

  absl::StatusOr<xls::dslx::TypecheckedModule> tm =
      xls::dslx::ParseAndTypecheck(text, path, module_name, cpp_import_data);
  if (tm.ok()) {
    auto* tm_on_heap = new xls::dslx::TypecheckedModule{*std::move(tm)};
    *result_out = reinterpret_cast<xls_dslx_typechecked_module*>(tm_on_heap);
    *error_out = nullptr;
    return true;
  }

  *result_out = nullptr;
  *error_out = xls::ToOwnedCString(tm.status().ToString());
  return false;
}

bool xls_dslx_typechecked_module_clone_removing_functions(
    struct xls_dslx_typechecked_module* tm,
    struct xls_dslx_function* functions[], size_t function_count,
    const char* install_subject, struct xls_dslx_import_data* import_data,
    char** error_out, struct xls_dslx_typechecked_module** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  auto fail = [&](absl::string_view message) {
    *error_out = xls::ToOwnedCString(std::string(message));
    *result_out = nullptr;
    return false;
  };
  if (function_count != 0 && functions == nullptr) {
    return fail("functions array is null");
  }
  std::vector<xls_dslx_module_member*> members;
  members.reserve(function_count);
  for (size_t i = 0; i < function_count; ++i) {
    auto* fn = reinterpret_cast<xls::dslx::Function*>(functions[i]);
    xls::dslx::ModuleMember* member = FindModuleMemberForNode(fn);
    if (member == nullptr) {
      return fail("function does not belong to the provided module");
    }
    members.push_back(reinterpret_cast<xls_dslx_module_member*>(member));
  }
  return xls_dslx_typechecked_module_clone_removing_members(
      tm, members.empty() ? nullptr : members.data(), function_count,
      install_subject, import_data, error_out, result_out);
}

bool xls_dslx_typechecked_module_clone_removing_members(
    struct xls_dslx_typechecked_module* tm,
    struct xls_dslx_module_member* members[], size_t member_count,
    const char* install_subject, struct xls_dslx_import_data* import_data,
    char** error_out, struct xls_dslx_typechecked_module** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  *error_out = nullptr;
  *result_out = nullptr;

  if (tm == nullptr || import_data == nullptr) {
    *error_out = xls::ToOwnedCString("null argument provided");
    return false;
  }
  if (member_count != 0 && members == nullptr) {
    *error_out = xls::ToOwnedCString("members array is null");
    return false;
  }

  auto* cpp_tm = reinterpret_cast<xls::dslx::TypecheckedModule*>(tm);
  auto* cpp_import_data = reinterpret_cast<xls::dslx::ImportData*>(import_data);

  std::string subject = std::string(install_subject);
  if (subject.empty()) {
    *error_out = xls::ToOwnedCString("install_subject must not be empty");
    return false;
  }

  std::vector<const xls::dslx::AstNode*> nodes_to_remove;
  nodes_to_remove.reserve(member_count);
  for (size_t i = 0; i < member_count; ++i) {
    if (members[i] == nullptr) {
      *error_out = xls::ToOwnedCString("members array contains null entry");
      return false;
    }
    auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(members[i]);
    const xls::dslx::AstNode* node = xls::dslx::ToAstNode(*cpp_member);
    if (node == nullptr || node->owner() != cpp_tm->module) {
      *error_out = xls::ToOwnedCString(
          "module member does not belong to the provided module");
      return false;
    }
    nodes_to_remove.push_back(node);
  }

  absl::StatusOr<std::unique_ptr<xls::dslx::Module>> cloned_module_or =
      xls::dslx::CloneModuleRemovingMembers(*cpp_tm->module, nodes_to_remove);
  if (!cloned_module_or.ok()) {
    *error_out = xls::ToOwnedCString(cloned_module_or.status().ToString());
    return false;
  }

  std::unique_ptr<xls::dslx::Module> cloned_module =
      std::move(cloned_module_or).value();
  if (cloned_module->name() != subject) {
    cloned_module->SetName(subject);
  }
  std::string path = cpp_tm->module->fs_path().has_value()
                         ? cpp_tm->module->fs_path()->string()
                         : std::string(cpp_tm->module->name());

  absl::StatusOr<xls::dslx::TypecheckedModule> retyped =
      xls::dslx::TypecheckModule(std::move(cloned_module), path,
                                 cpp_import_data);
  if (!retyped.ok()) {
    *error_out = xls::ToOwnedCString(retyped.status().ToString());
    return false;
  }
  auto* new_tm = new xls::dslx::TypecheckedModule{std::move(retyped).value()};
  *result_out = reinterpret_cast<xls_dslx_typechecked_module*>(new_tm);
  return true;
}

int64_t xls_dslx_module_get_member_count(struct xls_dslx_module* module) {
  CHECK(module != nullptr);
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  return cpp_module->top().size();
}

xls_dslx_module_member_kind xls_dslx_module_member_get_kind(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  xls::dslx::ModuleMember& cpp_member_ref = *cpp_member;
  xls_dslx_module_member_kind result = absl::visit(
      xls::Visitor{
          [](xls::dslx::Function*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_function;
          },
          [](xls::dslx::Proc*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_proc;
          },
          [](xls::dslx::ProcAlias*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_proc_alias;
          },
          [](xls::dslx::TestFunction*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_test_function;
          },
          [](xls::dslx::TestProc*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_test_proc;
          },
          [](xls::dslx::QuickCheck*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_quick_check;
          },
          [](xls::dslx::TypeAlias*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_type_alias;
          },
          [](xls::dslx::StructDef*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_struct_def;
          },
          [](xls::dslx::ProcDef*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_proc_def;
          },
          [](xls::dslx::ConstantDef*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_constant_def;
          },
          [](xls::dslx::EnumDef*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_enum_def;
          },
          [](xls::dslx::Import*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_import;
          },
          [](xls::dslx::Use*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_use;
          },
          [](xls::dslx::ConstAssert*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_const_assert;
          },
          [](xls::dslx::Impl*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_impl;
          },
          [](xls::dslx::Trait*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_trait;
          },
          [](xls::dslx::VerbatimNode*&) -> xls_dslx_module_member_kind {
            return xls_dslx_module_member_kind_verbatim_node;
          },
      },
      cpp_member_ref);
  return result;
}

struct xls_dslx_module_member* xls_dslx_module_get_member(
    struct xls_dslx_module* module, int64_t i) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  xls::dslx::ModuleMember& cpp_member = cpp_module->top().at(i);
  return reinterpret_cast<xls_dslx_module_member*>(&cpp_member);
}

struct xls_dslx_constant_def* xls_dslx_module_member_get_constant_def(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  auto* cpp_constant_def = std::get<xls::dslx::ConstantDef*>(*cpp_member);
  return reinterpret_cast<xls_dslx_constant_def*>(cpp_constant_def);
}

struct xls_dslx_struct_def* xls_dslx_module_member_get_struct_def(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  auto* cpp_struct_def = std::get<xls::dslx::StructDef*>(*cpp_member);
  return reinterpret_cast<xls_dslx_struct_def*>(cpp_struct_def);
}

struct xls_dslx_enum_def* xls_dslx_module_member_get_enum_def(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  auto* cpp_enum_def = std::get<xls::dslx::EnumDef*>(*cpp_member);
  return reinterpret_cast<xls_dslx_enum_def*>(cpp_enum_def);
}

struct xls_dslx_type_alias* xls_dslx_module_member_get_type_alias(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  auto* cpp_type_alias = std::get<xls::dslx::TypeAlias*>(*cpp_member);
  return reinterpret_cast<xls_dslx_type_alias*>(cpp_type_alias);
}

struct xls_dslx_function* xls_dslx_module_member_get_function(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  if (std::holds_alternative<xls::dslx::Function*>(*cpp_member)) {
    auto* cpp_function = std::get<xls::dslx::Function*>(*cpp_member);
    return reinterpret_cast<xls_dslx_function*>(cpp_function);
  }
  return nullptr;
}

struct xls_dslx_quickcheck* xls_dslx_module_member_get_quickcheck(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  auto* cpp_qc = std::get<xls::dslx::QuickCheck*>(*cpp_member);
  return reinterpret_cast<xls_dslx_quickcheck*>(cpp_qc);
}

struct xls_dslx_module_member* xls_dslx_module_member_from_constant_def(
    struct xls_dslx_constant_def* constant_def) {
  auto* cpp_constant_def =
      reinterpret_cast<xls::dslx::ConstantDef*>(constant_def);
  xls::dslx::ModuleMember* member = FindModuleMemberForNode(cpp_constant_def);
  return reinterpret_cast<xls_dslx_module_member*>(member);
}

struct xls_dslx_module_member* xls_dslx_module_member_from_struct_def(
    struct xls_dslx_struct_def* struct_def) {
  auto* cpp_struct_def = reinterpret_cast<xls::dslx::StructDef*>(struct_def);
  xls::dslx::ModuleMember* member = FindModuleMemberForNode(cpp_struct_def);
  return reinterpret_cast<xls_dslx_module_member*>(member);
}

struct xls_dslx_module_member* xls_dslx_module_member_from_enum_def(
    struct xls_dslx_enum_def* enum_def) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(enum_def);
  xls::dslx::ModuleMember* member = FindModuleMemberForNode(cpp_enum_def);
  return reinterpret_cast<xls_dslx_module_member*>(member);
}

struct xls_dslx_module_member* xls_dslx_module_member_from_type_alias(
    struct xls_dslx_type_alias* type_alias) {
  auto* cpp_type_alias = reinterpret_cast<xls::dslx::TypeAlias*>(type_alias);
  xls::dslx::ModuleMember* member = FindModuleMemberForNode(cpp_type_alias);
  return reinterpret_cast<xls_dslx_module_member*>(member);
}

struct xls_dslx_module_member* xls_dslx_module_member_from_function(
    struct xls_dslx_function* function) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(function);
  xls::dslx::ModuleMember* member = FindModuleMemberForNode(cpp_function);
  return reinterpret_cast<xls_dslx_module_member*>(member);
}

struct xls_dslx_module_member* xls_dslx_module_member_from_quickcheck(
    struct xls_dslx_quickcheck* quickcheck) {
  auto* cpp_qc = reinterpret_cast<xls::dslx::QuickCheck*>(quickcheck);
  xls::dslx::ModuleMember* member = FindModuleMemberForNode(cpp_qc);
  return reinterpret_cast<xls_dslx_module_member*>(member);
}

bool xls_dslx_function_is_parametric(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  return cpp_function->IsParametric();
}

bool xls_dslx_function_is_public(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  return cpp_function->is_public();
}

char* xls_dslx_function_get_identifier(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  const std::string& result = cpp_function->identifier();
  return xls::ToOwnedCString(result);
}

int64_t xls_dslx_function_get_param_count(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  return static_cast<int64_t>(cpp_function->params().size());
}

int64_t xls_dslx_function_get_parametric_binding_count(
    struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  return static_cast<int64_t>(cpp_function->parametric_bindings().size());
}

struct xls_dslx_param* xls_dslx_function_get_param(struct xls_dslx_function* fn,
                                                   int64_t index) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  xls::dslx::Param* cpp_param = cpp_function->params().at(index);
  return reinterpret_cast<xls_dslx_param*>(cpp_param);
}

struct xls_dslx_parametric_binding* xls_dslx_function_get_parametric_binding(
    struct xls_dslx_function* fn, int64_t index) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  xls::dslx::ParametricBinding* cpp_binding =
      cpp_function->parametric_bindings().at(index);
  return reinterpret_cast<xls_dslx_parametric_binding*>(cpp_binding);
}

char* xls_dslx_param_get_name(struct xls_dslx_param* p) {
  auto* cpp_param = reinterpret_cast<xls::dslx::Param*>(p);
  return xls::ToOwnedCString(cpp_param->name_def()->identifier());
}

struct xls_dslx_type_annotation* xls_dslx_param_get_type_annotation(
    struct xls_dslx_param* p) {
  auto* cpp_param = reinterpret_cast<xls::dslx::Param*>(p);
  xls::dslx::TypeAnnotation* cpp_ta = cpp_param->type_annotation();
  return reinterpret_cast<xls_dslx_type_annotation*>(cpp_ta);
}

struct xls_dslx_expr* xls_dslx_function_get_body(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  xls::dslx::StatementBlock* cpp_body = cpp_function->body();
  return reinterpret_cast<xls_dslx_expr*>(cpp_body);
}

struct xls_dslx_type_annotation* xls_dslx_function_get_return_type(
    struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  xls::dslx::TypeAnnotation* cpp_return_type = cpp_function->return_type();
  return reinterpret_cast<xls_dslx_type_annotation*>(cpp_return_type);
}

int64_t xls_dslx_function_get_attribute_count(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  return static_cast<int64_t>(cpp_function->attributes().size());
}

struct xls_dslx_attribute* xls_dslx_function_get_attribute(
    struct xls_dslx_function* fn, int64_t index) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  xls::dslx::Attribute* cpp_attribute = cpp_function->attributes().at(index);
  return reinterpret_cast<xls_dslx_attribute*>(cpp_attribute);
}

xls_dslx_attribute_kind xls_dslx_attribute_get_kind(
    struct xls_dslx_attribute* attribute) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  switch (cpp_attribute->attribute_kind()) {
    case xls::dslx::AttributeKind::kCfg:
      return xls_dslx_attribute_kind_cfg;
    case xls::dslx::AttributeKind::kDslxFormatDisable:
      return xls_dslx_attribute_kind_dslx_format_disable;
    case xls::dslx::AttributeKind::kExternVerilog:
      return xls_dslx_attribute_kind_extern_verilog;
    case xls::dslx::AttributeKind::kSvType:
      return xls_dslx_attribute_kind_sv_type;
    case xls::dslx::AttributeKind::kTest:
      return xls_dslx_attribute_kind_test;
    case xls::dslx::AttributeKind::kTestProc:
      return xls_dslx_attribute_kind_test_proc;
    case xls::dslx::AttributeKind::kQuickcheck:
      return xls_dslx_attribute_kind_quickcheck;
    default:
      CHECK(false) << "Unhandled attribute kind";
  }
  return xls_dslx_attribute_kind_cfg;
}

int64_t xls_dslx_attribute_get_argument_count(
    struct xls_dslx_attribute* attribute) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  return static_cast<int64_t>(cpp_attribute->args().size());
}

xls_dslx_attribute_argument_kind xls_dslx_attribute_get_argument_kind(
    struct xls_dslx_attribute* attribute, int64_t index) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  const xls::dslx::Attribute::Argument& argument =
      cpp_attribute->args().at(index);
  if (std::holds_alternative<std::string>(argument)) {
    return xls_dslx_attribute_argument_kind_string;
  }
  if (std::holds_alternative<xls::dslx::Attribute::StringKeyValueArgument>(
          argument)) {
    return xls_dslx_attribute_argument_kind_string_key_value;
  }
  if (std::holds_alternative<xls::dslx::Attribute::IntKeyValueArgument>(
          argument)) {
    return xls_dslx_attribute_argument_kind_int_key_value;
  }
  CHECK(false) << "Unexpected attribute argument kind";
  return xls_dslx_attribute_argument_kind_string;
}

char* xls_dslx_attribute_get_string_argument(
    struct xls_dslx_attribute* attribute, int64_t index) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  const xls::dslx::Attribute::Argument& argument =
      cpp_attribute->args().at(index);
  const std::string* value = std::get_if<std::string>(&argument);
  CHECK(value != nullptr) << "Attribute argument is not a string";
  return xls::ToOwnedCString(*value);
}

char* xls_dslx_attribute_get_key_value_argument_key(
    struct xls_dslx_attribute* attribute, int64_t index) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  const xls::dslx::Attribute::Argument& argument =
      cpp_attribute->args().at(index);
  if (const auto* kv =
          std::get_if<xls::dslx::Attribute::StringKeyValueArgument>(
              &argument)) {
    return xls::ToOwnedCString(kv->first);
  }
  if (const auto* kv =
          std::get_if<xls::dslx::Attribute::IntKeyValueArgument>(&argument)) {
    return xls::ToOwnedCString(kv->first);
  }
  CHECK(false) << "Attribute argument is not key/value";
  return nullptr;
}

char* xls_dslx_attribute_get_key_value_string_argument_value(
    struct xls_dslx_attribute* attribute, int64_t index) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  const xls::dslx::Attribute::Argument& argument =
      cpp_attribute->args().at(index);
  const auto* kv =
      std::get_if<xls::dslx::Attribute::StringKeyValueArgument>(&argument);
  CHECK(kv != nullptr)
      << "Attribute argument is not a string key/value argument";
  return xls::ToOwnedCString(kv->second);
}

int64_t xls_dslx_attribute_get_key_value_int_argument_value(
    struct xls_dslx_attribute* attribute, int64_t index) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  const xls::dslx::Attribute::Argument& argument =
      cpp_attribute->args().at(index);
  const auto* kv =
      std::get_if<xls::dslx::Attribute::IntKeyValueArgument>(&argument);
  CHECK(kv != nullptr) << "Attribute argument is not an int key/value argument";
  return kv->second;
}

char* xls_dslx_attribute_to_string(struct xls_dslx_attribute* attribute) {
  auto* cpp_attribute = reinterpret_cast<xls::dslx::Attribute*>(attribute);
  return xls::ToOwnedCString(cpp_attribute->ToString());
}

char* xls_dslx_parametric_binding_get_identifier(
    struct xls_dslx_parametric_binding* binding) {
  auto* cpp_binding = reinterpret_cast<xls::dslx::ParametricBinding*>(binding);
  return xls::ToOwnedCString(cpp_binding->identifier());
}

struct xls_dslx_type_annotation*
xls_dslx_parametric_binding_get_type_annotation(
    struct xls_dslx_parametric_binding* binding) {
  auto* cpp_binding = reinterpret_cast<xls::dslx::ParametricBinding*>(binding);
  xls::dslx::TypeAnnotation* cpp_type = cpp_binding->type_annotation();
  return reinterpret_cast<xls_dslx_type_annotation*>(cpp_type);
}

struct xls_dslx_expr* xls_dslx_parametric_binding_get_expr(
    struct xls_dslx_parametric_binding* binding) {
  auto* cpp_binding = reinterpret_cast<xls::dslx::ParametricBinding*>(binding);
  xls::dslx::Expr* cpp_expr = cpp_binding->expr();
  return reinterpret_cast<xls_dslx_expr*>(cpp_expr);
}

char* xls_dslx_function_to_string(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  return xls::ToOwnedCString(cpp_function->ToString());
}

char* xls_dslx_expr_to_string(struct xls_dslx_expr* expr) {
  auto* cpp_expr = reinterpret_cast<xls::dslx::Expr*>(expr);
  return xls::ToOwnedCString(cpp_expr->ToString());
}

bool xls_dslx_type_info_build_function_call_graph(
    struct xls_dslx_type_info* type_info, char** error_out,
    struct xls_dslx_call_graph** result_out) {
  CHECK(type_info != nullptr);
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  *error_out = nullptr;
  *result_out = nullptr;

  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto graph = cpp_type_info->GetFunctionCallGraph();
  auto holder = std::make_unique<CallGraphHolder>();
  holder->type_info = cpp_type_info;
  holder->graph = std::move(graph);

  xls::dslx::Module* module = cpp_type_info->module();
  for (xls::dslx::ModuleMember& member : module->top()) {
    if (std::holds_alternative<xls::dslx::Function*>(member)) {
      const xls::dslx::Function* fn = std::get<xls::dslx::Function*>(member);
      holder->functions.push_back(fn);
      if (!holder->graph.contains(fn)) {
        holder->graph.emplace(fn, std::vector<const xls::dslx::Function*>{});
      }
    }
  }

  *result_out = reinterpret_cast<xls_dslx_call_graph*>(holder.release());
  return true;
}

void xls_dslx_call_graph_free(struct xls_dslx_call_graph* call_graph) {
  delete reinterpret_cast<CallGraphHolder*>(call_graph);
}

int64_t xls_dslx_call_graph_get_function_count(
    struct xls_dslx_call_graph* call_graph) {
  if (call_graph == nullptr) {
    return 0;
  }
  auto* holder = reinterpret_cast<CallGraphHolder*>(call_graph);
  return static_cast<int64_t>(holder->functions.size());
}

struct xls_dslx_function* xls_dslx_call_graph_get_function(
    struct xls_dslx_call_graph* call_graph, int64_t index) {
  if (call_graph == nullptr) {
    return nullptr;
  }
  auto* holder = reinterpret_cast<CallGraphHolder*>(call_graph);
  if (index < 0 || index >= static_cast<int64_t>(holder->functions.size())) {
    return nullptr;
  }
  const xls::dslx::Function* fn = holder->functions.at(index);
  return reinterpret_cast<xls_dslx_function*>(
      const_cast<xls::dslx::Function*>(fn));
}

int64_t xls_dslx_call_graph_get_callee_count(
    struct xls_dslx_call_graph* call_graph, struct xls_dslx_function* caller) {
  if (call_graph == nullptr || caller == nullptr) {
    return 0;
  }
  auto* holder = reinterpret_cast<CallGraphHolder*>(call_graph);
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(caller);
  auto it = holder->graph.find(cpp_function);
  if (it == holder->graph.end()) {
    return 0;
  }
  return static_cast<int64_t>(it->second.size());
}

struct xls_dslx_function* xls_dslx_call_graph_get_callee_function(
    struct xls_dslx_call_graph* call_graph, struct xls_dslx_function* caller,
    int64_t callee_index) {
  if (call_graph == nullptr || caller == nullptr) {
    return nullptr;
  }
  auto* holder = reinterpret_cast<CallGraphHolder*>(call_graph);
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(caller);
  auto it = holder->graph.find(cpp_function);
  if (it == holder->graph.end()) {
    return nullptr;
  }
  const std::vector<const xls::dslx::Function*>& callees = it->second;
  if (callee_index < 0 ||
      callee_index >= static_cast<int64_t>(callees.size())) {
    return nullptr;
  }
  const xls::dslx::Function* fn = callees.at(callee_index);
  return reinterpret_cast<xls_dslx_function*>(
      const_cast<xls::dslx::Function*>(fn));
}

struct xls_dslx_function* xls_dslx_quickcheck_get_function(
    struct xls_dslx_quickcheck* quickcheck) {
  auto* cpp_qc = reinterpret_cast<xls::dslx::QuickCheck*>(quickcheck);
  xls::dslx::Function* cpp_fn = cpp_qc->fn();
  return reinterpret_cast<xls_dslx_function*>(cpp_fn);
}

bool xls_dslx_quickcheck_is_exhaustive(struct xls_dslx_quickcheck* quickcheck) {
  auto* cpp_qc = reinterpret_cast<xls::dslx::QuickCheck*>(quickcheck);
  return cpp_qc->test_cases().tag() ==
         xls::dslx::QuickCheckTestCasesTag::kExhaustive;
}

bool xls_dslx_quickcheck_get_count(struct xls_dslx_quickcheck* quickcheck,
                                   int64_t* result_out) {
  auto* cpp_qc = reinterpret_cast<xls::dslx::QuickCheck*>(quickcheck);
  const xls::dslx::QuickCheckTestCases& tc = cpp_qc->test_cases();
  if (tc.tag() != xls::dslx::QuickCheckTestCasesTag::kCounted) {
    return false;
  }
  std::optional<int64_t> count = tc.count();
  if (count.has_value()) {
    *result_out = *count;
  } else {
    *result_out = xls::dslx::QuickCheckTestCases::kDefaultTestCount;
  }
  return true;
}

char* xls_dslx_quickcheck_to_string(struct xls_dslx_quickcheck* quickcheck) {
  auto* cpp_qc = reinterpret_cast<xls::dslx::QuickCheck*>(quickcheck);
  return xls::ToOwnedCString(cpp_qc->ToString());
}

bool xls_dslx_type_info_get_requires_implicit_token(
    struct xls_dslx_type_info* type_info, struct xls_dslx_function* function,
    char** error_out, bool* result_out) {
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  CHECK(cpp_type_info != nullptr);
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(function);
  CHECK(cpp_function != nullptr);
  std::optional<bool> requires_implicit_token =
      cpp_type_info->GetRequiresImplicitToken(*cpp_function);
  if (!requires_implicit_token.has_value()) {
    *result_out = false;
    *error_out = xls::ToOwnedCString(
        absl::NotFoundError(
            "No implicit-token calling convention information for function")
            .ToString());
    return false;
  }

  *result_out = *requires_implicit_token;
  *error_out = nullptr;
  return true;
}

int64_t xls_dslx_module_get_type_definition_count(
    struct xls_dslx_module* module) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  return cpp_module->GetTypeDefinitions().size();
}

xls_dslx_type_definition_kind xls_dslx_module_get_type_definition_kind(
    struct xls_dslx_module* module, int64_t i) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  xls::dslx::TypeDefinition cpp_type_definition =
      cpp_module->GetTypeDefinitions().at(i);
  return absl::visit(xls::Visitor{
                         [](const xls::dslx::StructDef*) {
                           return xls_dslx_type_definition_kind_struct_def;
                         },
                         [](const xls::dslx::ProcDef*) {
                           return xls_dslx_type_definition_kind_proc_def;
                         },
                         [](const xls::dslx::EnumDef*) {
                           return xls_dslx_type_definition_kind_enum_def;
                         },
                         [](const xls::dslx::TypeAlias*) {
                           return xls_dslx_type_definition_kind_type_alias;
                         },
                         [](const xls::dslx::ColonRef*) {
                           return xls_dslx_type_definition_kind_colon_ref;
                         },
                         [](const xls::dslx::UseTreeEntry*) {
                           return xls_dslx_type_definition_kind_use_tree_entry;
                         },
                     },
                     cpp_type_definition);
}

char* xls_dslx_module_get_name(struct xls_dslx_module* module) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  const std::string& result = cpp_module->name();
  return xls::ToOwnedCString(result);
}

char* xls_dslx_module_to_string(struct xls_dslx_module* module) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  return xls::ToOwnedCString(cpp_module->ToString());
}

struct xls_dslx_struct_def* xls_dslx_module_get_type_definition_as_struct_def(
    struct xls_dslx_module* module, int64_t i) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  xls::dslx::TypeDefinition cpp_type_definition =
      cpp_module->GetTypeDefinitions().at(i);
  auto* cpp_struct_def = std::get<xls::dslx::StructDef*>(cpp_type_definition);
  return reinterpret_cast<xls_dslx_struct_def*>(cpp_struct_def);
}

struct xls_dslx_enum_def* xls_dslx_module_get_type_definition_as_enum_def(
    struct xls_dslx_module* module, int64_t i) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  xls::dslx::TypeDefinition cpp_type_definition =
      cpp_module->GetTypeDefinitions().at(i);
  auto* cpp_enum_def = std::get<xls::dslx::EnumDef*>(cpp_type_definition);
  return reinterpret_cast<xls_dslx_enum_def*>(cpp_enum_def);
}

struct xls_dslx_type_alias* xls_dslx_module_get_type_definition_as_type_alias(
    struct xls_dslx_module* module, int64_t i) {
  auto* cpp_module = reinterpret_cast<xls::dslx::Module*>(module);
  xls::dslx::TypeDefinition cpp_type_definition =
      cpp_module->GetTypeDefinitions().at(i);
  auto* cpp_type_alias = std::get<xls::dslx::TypeAlias*>(cpp_type_definition);
  return reinterpret_cast<xls_dslx_type_alias*>(cpp_type_alias);
}

char* xls_dslx_struct_def_get_identifier(struct xls_dslx_struct_def* n) {
  auto* cpp_struct_def = reinterpret_cast<xls::dslx::StructDef*>(n);
  const std::string& result = cpp_struct_def->identifier();
  return xls::ToOwnedCString(result);
}

bool xls_dslx_struct_def_is_parametric(struct xls_dslx_struct_def* n) {
  auto* cpp_struct_def = reinterpret_cast<xls::dslx::StructDef*>(n);
  return cpp_struct_def->IsParametric();
}

int64_t xls_dslx_struct_def_get_member_count(struct xls_dslx_struct_def* n) {
  auto* cpp_struct_def = reinterpret_cast<xls::dslx::StructDef*>(n);
  return cpp_struct_def->size();
}

char* xls_dslx_struct_def_to_string(struct xls_dslx_struct_def* n) {
  auto* cpp_struct_def = reinterpret_cast<xls::dslx::StructDef*>(n);
  return xls::ToOwnedCString(cpp_struct_def->ToString());
}

// -- colon_ref

struct xls_dslx_import* xls_dslx_colon_ref_resolve_import_subject(
    struct xls_dslx_colon_ref* n) {
  auto* cpp_colon_ref = reinterpret_cast<xls::dslx::ColonRef*>(n);
  std::optional<std::variant<xls::dslx::UseTreeEntry*, xls::dslx::Import*>>
      cpp_import = cpp_colon_ref->ResolveImportSubject();
  if (!cpp_import.has_value()) {
    return nullptr;
  }
  return absl::visit(
      xls::Visitor{
          [](xls::dslx::UseTreeEntry* entry) -> xls_dslx_import* {
            return nullptr;
          },
          [](xls::dslx::Import* import) -> xls_dslx_import* {
            return reinterpret_cast<xls_dslx_import*>(import);
          },
      },
      cpp_import.value());
}

char* xls_dslx_colon_ref_get_attr(struct xls_dslx_colon_ref* n) {
  auto* cpp_colon_ref = reinterpret_cast<xls::dslx::ColonRef*>(n);
  const std::string& result = cpp_colon_ref->attr();
  return xls::ToOwnedCString(result);
}

// -- type_alias

char* xls_dslx_type_alias_get_identifier(struct xls_dslx_type_alias* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeAlias*>(n);
  const std::string& result = cpp->identifier();
  return xls::ToOwnedCString(result);
}

struct xls_dslx_type_annotation* xls_dslx_type_alias_get_type_annotation(
    struct xls_dslx_type_alias* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeAlias*>(n);
  xls::dslx::TypeAnnotation& cpp_type_annotation = cpp->type_annotation();
  return reinterpret_cast<xls_dslx_type_annotation*>(&cpp_type_annotation);
}

char* xls_dslx_type_alias_to_string(struct xls_dslx_type_alias* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeAlias*>(n);
  return xls::ToOwnedCString(cpp->ToString());
}

// -- type_annotation

struct xls_dslx_type_ref_type_annotation*
xls_dslx_type_annotation_get_type_ref_type_annotation(
    struct xls_dslx_type_annotation* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeAnnotation*>(n);
  auto* cpp_type_ref = dynamic_cast<xls::dslx::TypeRefTypeAnnotation*>(cpp);
  return reinterpret_cast<xls_dslx_type_ref_type_annotation*>(cpp_type_ref);
}

// -- type_ref_type_annotation

struct xls_dslx_type_ref* xls_dslx_type_ref_type_annotation_get_type_ref(
    struct xls_dslx_type_ref_type_annotation* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeRefTypeAnnotation*>(n);
  auto* cpp_type_ref = cpp->type_ref();
  return reinterpret_cast<xls_dslx_type_ref*>(cpp_type_ref);
}

// -- type_ref

struct xls_dslx_type_definition* xls_dslx_type_ref_get_type_definition(
    struct xls_dslx_type_ref* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeRef*>(n);
  // const_cast is ok because the C API can only do immutable query-like things
  // with the node anyway.
  auto& cpp_type_def =
      const_cast<xls::dslx::TypeDefinition&>(cpp->type_definition());
  return reinterpret_cast<xls_dslx_type_definition*>(&cpp_type_def);
}

// -- type_definition

struct xls_dslx_colon_ref* xls_dslx_type_definition_get_colon_ref(
    struct xls_dslx_type_definition* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeDefinition*>(n);
  if (std::holds_alternative<xls::dslx::ColonRef*>(*cpp)) {
    auto* colon_ref = std::get<xls::dslx::ColonRef*>(*cpp);
    return reinterpret_cast<xls_dslx_colon_ref*>(colon_ref);
  }
  return nullptr;
}

struct xls_dslx_type_alias* xls_dslx_type_definition_get_type_alias(
    struct xls_dslx_type_definition* n) {
  auto* cpp = reinterpret_cast<xls::dslx::TypeDefinition*>(n);
  if (std::holds_alternative<xls::dslx::TypeAlias*>(*cpp)) {
    auto* type_alias = std::get<xls::dslx::TypeAlias*>(*cpp);
    return reinterpret_cast<xls_dslx_type_alias*>(type_alias);
  }
  return nullptr;
}

// -- import

int64_t xls_dslx_import_get_subject_count(struct xls_dslx_import* n) {
  auto* cpp = reinterpret_cast<xls::dslx::Import*>(n);
  return static_cast<int64_t>(cpp->subject().size());
}

char* xls_dslx_import_get_subject(struct xls_dslx_import* n, int64_t i) {
  auto* cpp = reinterpret_cast<xls::dslx::Import*>(n);
  const std::string& result = cpp->subject().at(i);
  return xls::ToOwnedCString(result);
}

// -- constant_def

char* xls_dslx_constant_def_get_name(struct xls_dslx_constant_def* n) {
  auto* cpp = reinterpret_cast<xls::dslx::ConstantDef*>(n);
  const std::string& result = cpp->name_def()->identifier();
  return xls::ToOwnedCString(result);
}

struct xls_dslx_expr* xls_dslx_constant_def_get_value(
    struct xls_dslx_constant_def* n) {
  auto* cpp = reinterpret_cast<xls::dslx::ConstantDef*>(n);
  xls::dslx::Expr* cpp_value = cpp->value();
  return reinterpret_cast<xls_dslx_expr*>(cpp_value);
}

char* xls_dslx_constant_def_to_string(struct xls_dslx_constant_def* n) {
  auto* cpp = reinterpret_cast<xls::dslx::ConstantDef*>(n);
  return xls::ToOwnedCString(cpp->ToString());
}

// -- enum_def

char* xls_dslx_enum_def_get_identifier(struct xls_dslx_enum_def* n) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  const std::string& result = cpp_enum_def->identifier();
  return xls::ToOwnedCString(result);
}

struct xls_dslx_type_annotation* xls_dslx_enum_def_get_underlying(
    struct xls_dslx_enum_def* n) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  auto* cpp_type_annotation = cpp_enum_def->type_annotation();
  return reinterpret_cast<xls_dslx_type_annotation*>(cpp_type_annotation);
}

int64_t xls_dslx_enum_def_get_member_count(struct xls_dslx_enum_def* n) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  return static_cast<int64_t>(cpp_enum_def->values().size());
}

struct xls_dslx_enum_member* xls_dslx_enum_def_get_member(
    struct xls_dslx_enum_def* n, int64_t i) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  xls::dslx::EnumMember& cpp_member = cpp_enum_def->mutable_values().at(i);
  return reinterpret_cast<xls_dslx_enum_member*>(&cpp_member);
}

char* xls_dslx_enum_member_get_name(struct xls_dslx_enum_member* m) {
  auto* cpp_member = reinterpret_cast<xls::dslx::EnumMember*>(m);
  return xls::ToOwnedCString(cpp_member->name_def->identifier());
}

struct xls_dslx_expr* xls_dslx_enum_member_get_value(
    struct xls_dslx_enum_member* m) {
  auto* cpp_member = reinterpret_cast<xls::dslx::EnumMember*>(m);
  xls::dslx::Expr* cpp_value = cpp_member->value;
  return reinterpret_cast<xls_dslx_expr*>(cpp_value);
}

char* xls_dslx_enum_def_to_string(struct xls_dslx_enum_def* n) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  return xls::ToOwnedCString(cpp_enum_def->ToString());
}

struct xls_dslx_module* xls_dslx_expr_get_owner_module(
    struct xls_dslx_expr* expr) {
  auto* cpp_expr = reinterpret_cast<xls::dslx::Expr*>(expr);
  xls::dslx::Module* cpp_module = cpp_expr->owner();
  return reinterpret_cast<xls_dslx_module*>(cpp_module);
}

// -- type_info

const struct xls_dslx_type* xls_dslx_type_info_get_type_struct_def(
    struct xls_dslx_type_info* type_info,
    struct xls_dslx_struct_def* struct_def) {
  auto* node = reinterpret_cast<xls::dslx::AstNode*>(struct_def);
  return GetMetaTypeHelper(type_info, node);
}

const struct xls_dslx_type* xls_dslx_type_info_get_type_struct_member(
    struct xls_dslx_type_info* type_info,
    struct xls_dslx_struct_member* struct_member) {
  // Note: StructMember is not itself an AST node, it's just a POD struct, so
  // we need to traverse to its type annotation.
  auto* cpp_struct_member =
      reinterpret_cast<xls::dslx::StructMember*>(struct_member);
  xls::dslx::TypeAnnotation* node = cpp_struct_member->type;
  return GetMetaTypeHelper(type_info, node);
}

const struct xls_dslx_type* xls_dslx_type_info_get_type_enum_def(
    struct xls_dslx_type_info* type_info, struct xls_dslx_enum_def* enum_def) {
  auto* node = reinterpret_cast<xls::dslx::AstNode*>(enum_def);
  return GetMetaTypeHelper(type_info, node);
}

const struct xls_dslx_type* xls_dslx_type_info_get_type_constant_def(
    struct xls_dslx_type_info* type_info,
    struct xls_dslx_constant_def* constant_def) {
  auto* cpp_node = reinterpret_cast<xls::dslx::AstNode*>(constant_def);
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  std::optional<xls::dslx::Type*> maybe_type = cpp_type_info->GetItem(cpp_node);
  if (!maybe_type.has_value()) {
    return nullptr;
  }
  CHECK(maybe_type.value() != nullptr);
  xls::dslx::Type* cpp_type = maybe_type.value();
  return reinterpret_cast<const struct xls_dslx_type*>(cpp_type);
}

const struct xls_dslx_type* xls_dslx_type_info_get_type_type_annotation(
    struct xls_dslx_type_info* type_info,
    struct xls_dslx_type_annotation* type_annotation) {
  auto* node = reinterpret_cast<xls::dslx::AstNode*>(type_annotation);
  return GetMetaTypeHelper(type_info, node);
}

bool xls_dslx_type_get_total_bit_count(const struct xls_dslx_type* type,
                                       char** error_out, int64_t* result_out) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  absl::StatusOr<xls::dslx::TypeDim> bit_count = cpp_type->GetTotalBitCount();
  if (!bit_count.ok()) {
    *result_out = 0;
    *error_out = xls::ToOwnedCString(bit_count.status().ToString());
    return false;
  }

  absl::StatusOr<int64_t> width_or = bit_count->GetAsInt64();
  if (!width_or.ok()) {
    *result_out = 0;
    *error_out = xls::ToOwnedCString(width_or.status().ToString());
    return false;
  }

  *result_out = width_or.value();
  *error_out = nullptr;
  return true;
}

struct xls_dslx_struct_member* xls_dslx_struct_def_get_member(
    struct xls_dslx_struct_def* struct_def, int64_t i) {
  auto* cpp_struct_def = reinterpret_cast<xls::dslx::StructDef*>(struct_def);
  xls::dslx::StructMember& cpp_member = cpp_struct_def->mutable_members().at(i);
  return reinterpret_cast<xls_dslx_struct_member*>(&cpp_member);
}

struct xls_dslx_type_annotation* xls_dslx_struct_member_get_type(
    struct xls_dslx_struct_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::StructMember*>(member);
  xls::dslx::TypeAnnotation* cpp_type_annotation = cpp_member->type;
  return reinterpret_cast<xls_dslx_type_annotation*>(cpp_type_annotation);
}

char* xls_dslx_struct_member_get_name(struct xls_dslx_struct_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::StructMember*>(member);
  const std::string& name = cpp_member->name;
  return xls::ToOwnedCString(name);
}

bool xls_dslx_type_info_get_const_expr(
    struct xls_dslx_type_info* type_info, struct xls_dslx_expr* expr,
    char** error_out, struct xls_dslx_interp_value** result_out) {
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto* cpp_expr = reinterpret_cast<xls::dslx::Expr*>(expr);
  absl::StatusOr<xls::dslx::InterpValue> value =
      cpp_type_info->GetConstExpr(cpp_expr);
  if (!value.ok()) {
    *result_out = nullptr;
    *error_out = xls::ToOwnedCString(value.status().ToString());
    return false;
  }

  auto* heap = new xls::dslx::InterpValue{*std::move(value)};
  *result_out = reinterpret_cast<xls_dslx_interp_value*>(heap);
  *error_out = nullptr;
  return true;
}

struct xls_dslx_invocation_callee_data_array*
xls_dslx_type_info_get_unique_invocation_callee_data(
    struct xls_dslx_type_info* type_info, struct xls_dslx_function* function) {
  CHECK(type_info != nullptr);
  CHECK(function != nullptr);
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(function);
  std::vector<xls::dslx::InvocationCalleeData> entries =
      cpp_type_info->GetUniqueInvocationCalleeData(cpp_function);
  auto* array = new InvocationCalleeDataArray(std::move(entries));
  return reinterpret_cast<xls_dslx_invocation_callee_data_array*>(array);
}

struct xls_dslx_invocation_callee_data_array*
xls_dslx_type_info_get_all_invocation_callee_data(
    struct xls_dslx_type_info* type_info, struct xls_dslx_function* function) {
  CHECK(type_info != nullptr);
  CHECK(function != nullptr);
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(function);
  std::vector<xls::dslx::InvocationCalleeData> entries =
      cpp_type_info->GetAllInvocationCalleeData(cpp_function);
  auto* array = new InvocationCalleeDataArray(std::move(entries));
  return reinterpret_cast<xls_dslx_invocation_callee_data_array*>(array);
}

struct xls_dslx_invocation_data* xls_dslx_type_info_get_root_invocation_data(
    struct xls_dslx_type_info* type_info,
    struct xls_dslx_invocation* invocation) {
  CHECK(type_info != nullptr);
  CHECK(invocation != nullptr);
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto* cpp_invocation = reinterpret_cast<xls::dslx::Invocation*>(invocation);
  std::optional<const xls::dslx::InvocationData*> result =
      cpp_type_info->GetInvocationData(cpp_invocation);
  if (!result.has_value()) {
    return nullptr;
  }
  return reinterpret_cast<xls_dslx_invocation_data*>(
      const_cast<xls::dslx::InvocationData*>(*result));
}

void xls_dslx_invocation_callee_data_array_free(
    struct xls_dslx_invocation_callee_data_array* array) {
  if (array == nullptr) {
    return;
  }
  auto* cpp_array = reinterpret_cast<InvocationCalleeDataArray*>(array);
  delete cpp_array;
}

int64_t xls_dslx_invocation_callee_data_array_get_count(
    struct xls_dslx_invocation_callee_data_array* array) {
  CHECK(array != nullptr);
  auto* cpp_array = reinterpret_cast<InvocationCalleeDataArray*>(array);
  return cpp_array->entries.size();
}

struct xls_dslx_invocation_callee_data*
xls_dslx_invocation_callee_data_array_get(
    struct xls_dslx_invocation_callee_data_array* array, int64_t index) {
  CHECK(array != nullptr);
  auto* cpp_array = reinterpret_cast<InvocationCalleeDataArray*>(array);
  xls::dslx::InvocationCalleeData& entry = cpp_array->entries.at(index);
  return reinterpret_cast<xls_dslx_invocation_callee_data*>(&entry);
}

struct xls_dslx_invocation_callee_data* xls_dslx_invocation_callee_data_clone(
    struct xls_dslx_invocation_callee_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationCalleeData*>(data);
  auto* clone = new xls::dslx::InvocationCalleeData(*cpp_data);
  return reinterpret_cast<xls_dslx_invocation_callee_data*>(clone);
}

void xls_dslx_invocation_callee_data_free(
    struct xls_dslx_invocation_callee_data* data) {
  if (data == nullptr) {
    return;
  }
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationCalleeData*>(data);
  delete cpp_data;
}

const struct xls_dslx_parametric_env*
xls_dslx_invocation_callee_data_get_callee_bindings(
    struct xls_dslx_invocation_callee_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationCalleeData*>(data);
  return reinterpret_cast<const struct xls_dslx_parametric_env*>(
      &cpp_data->callee_bindings);
}

const struct xls_dslx_parametric_env*
xls_dslx_invocation_callee_data_get_caller_bindings(
    struct xls_dslx_invocation_callee_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationCalleeData*>(data);
  return reinterpret_cast<const struct xls_dslx_parametric_env*>(
      &cpp_data->caller_bindings);
}

struct xls_dslx_type_info*
xls_dslx_invocation_callee_data_get_derived_type_info(
    struct xls_dslx_invocation_callee_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationCalleeData*>(data);
  return reinterpret_cast<xls_dslx_type_info*>(cpp_data->derived_type_info);
}

struct xls_dslx_invocation* xls_dslx_invocation_callee_data_get_invocation(
    struct xls_dslx_invocation_callee_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationCalleeData*>(data);
  return reinterpret_cast<xls_dslx_invocation*>(
      const_cast<xls::dslx::Invocation*>(cpp_data->invocation));
}

struct xls_dslx_invocation* xls_dslx_invocation_data_get_invocation(
    struct xls_dslx_invocation_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationData*>(data);
  return reinterpret_cast<xls_dslx_invocation*>(
      const_cast<xls::dslx::Invocation*>(cpp_data->node()));
}

struct xls_dslx_function* xls_dslx_invocation_data_get_callee(
    struct xls_dslx_invocation_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationData*>(data);
  return reinterpret_cast<xls_dslx_function*>(
      const_cast<xls::dslx::Function*>(cpp_data->callee()));
}

struct xls_dslx_function* xls_dslx_invocation_data_get_caller(
    struct xls_dslx_invocation_data* data) {
  CHECK(data != nullptr);
  auto* cpp_data = reinterpret_cast<xls::dslx::InvocationData*>(data);
  return reinterpret_cast<xls_dslx_function*>(
      const_cast<xls::dslx::Function*>(cpp_data->caller()));
}

// -- interp_value

char* xls_dslx_interp_value_to_string(struct xls_dslx_interp_value* v) {
  auto* cpp_interp_value = reinterpret_cast<xls::dslx::InterpValue*>(v);
  return xls::ToOwnedCString(cpp_interp_value->ToString());
}

void xls_dslx_interp_value_free(struct xls_dslx_interp_value* v) {
  auto* cpp_interp_value = reinterpret_cast<xls::dslx::InterpValue*>(v);
  delete cpp_interp_value;
}

bool xls_dslx_interp_value_convert_to_ir(const struct xls_dslx_interp_value* v,
                                         char** error_out,
                                         struct xls_value** result_out) {
  const auto* cpp_interp_value =
      reinterpret_cast<const xls::dslx::InterpValue*>(v);
  absl::StatusOr<xls::Value> ir_value = cpp_interp_value->ConvertToIr();
  if (!ir_value.ok()) {
    *error_out = xls::ToOwnedCString(ir_value.status().ToString());
    *result_out = nullptr;
    return false;
  }

  auto* heap = new xls::Value{*std::move(ir_value)};
  *result_out = reinterpret_cast<xls_value*>(heap);
  *error_out = nullptr;
  return true;
}

// -- type

bool xls_dslx_type_is_signed_bits(const struct xls_dslx_type* type,
                                  char** error_out, bool* result_out) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  absl::StatusOr<bool> is_signed = xls::dslx::IsSigned(*cpp_type);
  if (!is_signed.ok()) {
    *error_out = xls::ToOwnedCString(is_signed.status().ToString());
    *result_out = false;
    return false;
  }

  *error_out = nullptr;
  *result_out = *is_signed;
  return true;
}

bool xls_dslx_type_is_enum(const struct xls_dslx_type* type) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  return cpp_type->IsEnum();
}

bool xls_dslx_type_is_struct(const struct xls_dslx_type* type) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  return cpp_type->IsStruct();
}

bool xls_dslx_type_is_array(const struct xls_dslx_type* type) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  return cpp_type->IsArray();
}

struct xls_dslx_type* xls_dslx_type_array_get_element_type(
    struct xls_dslx_type* type) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  CHECK(cpp_type->IsArray());
  const xls::dslx::Type& cpp_element_type = cpp_type->AsArray().element_type();
  const auto* element_type =
      reinterpret_cast<const xls_dslx_type*>(&cpp_element_type);
  // const_cast is ok because the C API can only do immutable query-like things
  // with the type anyway.
  return const_cast<xls_dslx_type*>(element_type);
}

struct xls_dslx_type_dim* xls_dslx_type_array_get_size(
    struct xls_dslx_type* type) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  CHECK(cpp_type->IsArray());
  const xls::dslx::TypeDim& cpp_size = cpp_type->AsArray().size();
  auto* cpp_type_dim = new xls::dslx::TypeDim(cpp_size);
  return reinterpret_cast<xls_dslx_type_dim*>(cpp_type_dim);
}

struct xls_dslx_enum_def* xls_dslx_type_get_enum_def(
    struct xls_dslx_type* type) {
  auto* cpp_type = reinterpret_cast<xls::dslx::Type*>(type);
  CHECK(cpp_type->IsEnum());
  const xls::dslx::EnumType& enum_type = cpp_type->AsEnum();
  const xls::dslx::EnumDef& cpp_enum_def = enum_type.nominal_type();
  const auto* enum_def =
      reinterpret_cast<const xls_dslx_enum_def*>(&cpp_enum_def);
  // const_cast is ok because the C API can only do immutable query-like things
  // with the node anyway.
  return const_cast<xls_dslx_enum_def*>(enum_def);
}

struct xls_dslx_struct_def* xls_dslx_type_get_struct_def(
    struct xls_dslx_type* type) {
  auto* cpp_type = reinterpret_cast<xls::dslx::Type*>(type);
  CHECK(cpp_type->IsStruct());
  const xls::dslx::StructType& struct_type = cpp_type->AsStruct();
  const xls::dslx::StructDef& cpp_struct_def = struct_type.nominal_type();
  const auto* struct_def =
      reinterpret_cast<const xls_dslx_struct_def*>(&cpp_struct_def);
  // const_cast is ok because the C API can only do immutable query-like things
  // with the node anyway.
  return const_cast<xls_dslx_struct_def*>(struct_def);
}

bool xls_dslx_type_to_string(const struct xls_dslx_type* type, char** error_out,
                             char** result_out) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  *error_out = nullptr;
  *result_out = xls::ToOwnedCString(cpp_type->ToString());
  return true;
}

bool xls_dslx_type_is_bits_like(struct xls_dslx_type* type,
                                struct xls_dslx_type_dim** is_signed,
                                struct xls_dslx_type_dim** size) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  std::optional<xls::dslx::BitsLikeProperties> properties =
      GetBitsLike(*cpp_type);
  if (!properties.has_value()) {
    *is_signed = nullptr;
    *size = nullptr;
    return false;
  }

  *is_signed = reinterpret_cast<xls_dslx_type_dim*>(
      new xls::dslx::TypeDim(std::move(properties->is_signed)));
  *size = reinterpret_cast<xls_dslx_type_dim*>(
      new xls::dslx::TypeDim(std::move(properties->size)));
  return true;
}

// -- type_dim

bool xls_dslx_type_dim_is_parametric(struct xls_dslx_type_dim* td) {
  auto* cpp_type_dim = reinterpret_cast<xls::dslx::TypeDim*>(td);
  return cpp_type_dim->IsParametric();
}

bool xls_dslx_type_dim_get_as_bool(struct xls_dslx_type_dim* td,
                                   char** error_out, bool* result_out) {
  auto* cpp_type_dim = reinterpret_cast<xls::dslx::TypeDim*>(td);
  absl::StatusOr<bool> value = cpp_type_dim->GetAsBool();
  if (!value.ok()) {
    *result_out = false;
    *error_out = xls::ToOwnedCString(value.status().ToString());
    return false;
  }

  *result_out = *value;
  *error_out = nullptr;
  return true;
}

bool xls_dslx_type_dim_get_as_int64(struct xls_dslx_type_dim* td,
                                    char** error_out, int64_t* result_out) {
  auto* cpp_type_dim = reinterpret_cast<xls::dslx::TypeDim*>(td);
  absl::StatusOr<int64_t> value = cpp_type_dim->GetAsInt64();
  if (!value.ok()) {
    *result_out = 0;
    *error_out = xls::ToOwnedCString(value.status().ToString());
    return false;
  }

  *result_out = *value;
  *error_out = nullptr;
  return true;
}

void xls_dslx_type_dim_free(struct xls_dslx_type_dim* td) {
  auto* cpp_type_dim = reinterpret_cast<xls::dslx::TypeDim*>(td);
  delete cpp_type_dim;
}

bool xls_dslx_replace_invocations_in_module(
    struct xls_dslx_typechecked_module* tm,
    struct xls_dslx_function* const callers[], size_t callers_count,
    const struct xls_dslx_invocation_rewrite_rule* rules, size_t rules_count,
    struct xls_dslx_import_data* import_data, const char* install_subject,
    char** error_out, struct xls_dslx_typechecked_module** result_out) {
  CHECK(error_out != nullptr);
  CHECK(result_out != nullptr);
  *error_out = nullptr;
  *result_out = nullptr;
  CHECK(tm != nullptr);
  CHECK(import_data != nullptr);
  CHECK(install_subject != nullptr);
  CHECK(callers != nullptr || callers_count == 0);
  CHECK(rules != nullptr || rules_count == 0);

  auto* cpp_tm = reinterpret_cast<xls::dslx::TypecheckedModule*>(tm);
  auto* cpp_import_data = reinterpret_cast<xls::dslx::ImportData*>(import_data);

  std::vector<const xls::dslx::Function*> callers_cpp;
  callers_cpp.reserve(callers_count);
  for (size_t i = 0; i < callers_count; ++i) {
    CHECK(callers[i] != nullptr);
    callers_cpp.push_back(
        reinterpret_cast<const xls::dslx::Function*>(callers[i]));
  }

  std::vector<xls::dslx::InvocationRewriteRule> rules_cpp;
  rules_cpp.reserve(rules_count);
  for (size_t i = 0; i < rules_count; ++i) {
    const xls_dslx_invocation_rewrite_rule& r = rules[i];
    CHECK(r.from_callee != nullptr);
    CHECK(r.to_callee != nullptr);
    xls::dslx::InvocationRewriteRule rr;
    rr.from_callee =
        reinterpret_cast<const xls::dslx::Function*>(r.from_callee);
    rr.to_callee = reinterpret_cast<const xls::dslx::Function*>(r.to_callee);
    if (r.match_callee_env != nullptr) {
      rr.match_callee_env = *reinterpret_cast<const xls::dslx::ParametricEnv*>(
          r.match_callee_env);
    }
    if (r.to_callee_env != nullptr) {
      rr.to_callee_env =
          *reinterpret_cast<const xls::dslx::ParametricEnv*>(r.to_callee_env);
    }
    rules_cpp.push_back(std::move(rr));
  }

  absl::StatusOr<xls::dslx::TypecheckedModule> new_tm =
      xls::dslx::ReplaceInvocationsInModule(
          *cpp_tm, absl::MakeSpan(callers_cpp), absl::MakeSpan(rules_cpp),
          *cpp_import_data, std::string_view{install_subject});
  if (!new_tm.ok()) {
    *error_out = xls::ToOwnedCString(new_tm.status().ToString());
    return false;
  }
  auto* heap_tm = new xls::dslx::TypecheckedModule{*std::move(new_tm)};
  *result_out = reinterpret_cast<xls_dslx_typechecked_module*>(heap_tm);
  return true;
}

}  // extern "C"
