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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "xls/common/visitor.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/value.h"
#include "xls/public/c_api_impl_helpers.h"

namespace {
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
}  // namespace

extern "C" {

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
  auto* cpp_function = std::get<xls::dslx::Function*>(*cpp_member);
  return reinterpret_cast<xls_dslx_function*>(cpp_function);
}

bool xls_dslx_function_is_parametric(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  return cpp_function->IsParametric();
}

char* xls_dslx_function_get_identifier(struct xls_dslx_function* fn) {
  auto* cpp_function = reinterpret_cast<xls::dslx::Function*>(fn);
  const std::string& result = cpp_function->identifier();
  return xls::ToOwnedCString(result);
}

struct xls_dslx_quickcheck* xls_dslx_module_member_get_quickcheck(
    struct xls_dslx_module_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::ModuleMember*>(member);
  auto* cpp_qc = std::get<xls::dslx::QuickCheck*>(*cpp_member);
  return reinterpret_cast<xls_dslx_quickcheck*>(cpp_qc);
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

// -- interp_value

char* xls_dslx_interp_value_to_string(struct xls_dslx_interp_value* v) {
  auto* cpp_interp_value = reinterpret_cast<xls::dslx::InterpValue*>(v);
  return xls::ToOwnedCString(cpp_interp_value->ToString());
}

void xls_dslx_interp_value_free(struct xls_dslx_interp_value* v) {
  auto* cpp_interp_value = reinterpret_cast<xls::dslx::InterpValue*>(v);
  delete cpp_interp_value;
}

bool xls_dslx_interp_value_convert_to_ir(struct xls_dslx_interp_value* v,
                                         char** error_out,
                                         struct xls_value** result_out) {
  auto* cpp_interp_value = reinterpret_cast<xls::dslx::InterpValue*>(v);
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

}  // extern "C"
