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

#include "absl/types/variant.h"
#include "xls/common/visitor.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_kind.h"
#include "xls/public/c_api_impl_helpers.h"

namespace {
template <typename T>
inline const struct xls_dslx_type* GetMetaTypeHelper(
    struct xls_dslx_type_info* type_info, T* c_node) {
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto* cpp_node = reinterpret_cast<xls::dslx::AstNode*>(c_node);
  std::optional<xls::dslx::Type*> maybe_type = cpp_type_info->GetItem(cpp_node);
  if (!maybe_type.has_value()) {
    return nullptr;
  }
  CHECK(maybe_type.value() != nullptr);
  // Should always have a metatype as its associated type.
  absl::StatusOr<const xls::dslx::Type*> unwrapped_or =
      xls::dslx::UnwrapMetaType(*maybe_type.value());
  CHECK(unwrapped_or.ok());
  return reinterpret_cast<const struct xls_dslx_type*>(unwrapped_or.value());
}
}  // namespace

extern "C" {

struct xls_dslx_import_data* xls_dslx_import_data_create(
    const char* dslx_stdlib_path, const char* additional_search_paths[],
    size_t additional_search_paths_count) {
  std::filesystem::path cpp_stdlib_path{dslx_stdlib_path};
  std::vector<std::filesystem::path> cpp_additional_search_paths =
      xls::ToCpp(additional_search_paths, additional_search_paths_count);
  xls::dslx::ImportData import_data = CreateImportData(
      cpp_stdlib_path, cpp_additional_search_paths, xls::dslx::kAllWarningsSet);
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

  absl::StatusOr<xls::dslx::TypecheckedModule> tm_or =
      xls::dslx::ParseAndTypecheck(text, path, module_name, cpp_import_data);
  if (tm_or.ok()) {
    auto* tm_on_heap =
        new xls::dslx::TypecheckedModule{std::move(tm_or).value()};
    *result_out = reinterpret_cast<xls_dslx_typechecked_module*>(tm_on_heap);
    *error_out = nullptr;
    return true;
  }

  *result_out = nullptr;
  *error_out = xls::ToOwnedCString(tm_or.status().ToString());
  return false;
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
                         [](const xls::dslx::EnumDef*) {
                           return xls_dslx_type_definition_kind_enum_def;
                         },
                         [](const xls::dslx::TypeAlias*) {
                           return xls_dslx_type_definition_kind_type_alias;
                         },
                         [](const xls::dslx::ColonRef*) {
                           return xls_dslx_type_definition_kind_colon_ref;
                         },
                     },
                     cpp_type_definition);
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

char* xls_dslx_struct_def_get_identifier(struct xls_dslx_struct_def* n) {
  auto* cpp_struct_def = reinterpret_cast<xls::dslx::StructDef*>(n);
  std::string result = cpp_struct_def->identifier();
  return xls::ToOwnedCString(result);
}

// -- enum_def

char* xls_dslx_enum_def_get_identifier(struct xls_dslx_enum_def* n) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  std::string result = cpp_enum_def->identifier();
  return xls::ToOwnedCString(result);
}

int64_t xls_dslx_enum_def_get_member_count(struct xls_dslx_enum_def* n) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  return static_cast<int64_t>(cpp_enum_def->values().size());
}

struct xls_dslx_enum_member* xls_dslx_enum_def_get_member(
    struct xls_dslx_enum_def* n, int64_t i) {
  auto* cpp_enum_def = reinterpret_cast<xls::dslx::EnumDef*>(n);
  xls::dslx::EnumMember& cpp_member = cpp_enum_def->values().at(i);
  return reinterpret_cast<xls_dslx_enum_member*>(&cpp_member);
}

char* xls_dslx_enum_member_get_name(struct xls_dslx_enum_member* m) {
  auto* cpp_member = reinterpret_cast<xls::dslx::EnumMember*>(m);
  return xls::ToOwnedCString(cpp_member->name_def->identifier());
}

struct xls_dslx_expr* xls_dslx_enum_member_get_value(struct xls_dslx_enum_member* m) {
  auto* cpp_member = reinterpret_cast<xls::dslx::EnumMember*>(m);
  xls::dslx::Expr* cpp_value = cpp_member->value;
  return reinterpret_cast<xls_dslx_expr*>(cpp_value);
}

// -- type_info

const struct xls_dslx_type* xls_dslx_type_info_get_type_struct_def(
    struct xls_dslx_type_info* type_info,
    struct xls_dslx_struct_def* struct_def) {
  return GetMetaTypeHelper(type_info, struct_def);
}

const struct xls_dslx_type* xls_dslx_type_info_get_type_enum_def(
    struct xls_dslx_type_info* type_info, struct xls_dslx_enum_def* enum_def) {
  return GetMetaTypeHelper(type_info, enum_def);
}

const struct xls_dslx_type* xls_dslx_type_info_get_type_type_annotation(
    struct xls_dslx_type_info* type_info,
    struct xls_dslx_type_annotation* type_annotation) {
  return GetMetaTypeHelper(type_info, type_annotation);
}

bool xls_dslx_type_get_total_bit_count(const struct xls_dslx_type* type,
                                       char** error_out, int64_t* result_out) {
  const auto* cpp_type = reinterpret_cast<const xls::dslx::Type*>(type);
  absl::StatusOr<xls::dslx::TypeDim> bit_count_or =
      cpp_type->GetTotalBitCount();
  if (!bit_count_or.ok()) {
    *result_out = 0;
    *error_out = xls::ToOwnedCString(bit_count_or.status().ToString());
    return false;
  }

  absl::StatusOr<int64_t> width_or = bit_count_or->GetAsInt64();
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
  xls::dslx::StructMember& cpp_member = cpp_struct_def->members().at(i);
  return reinterpret_cast<xls_dslx_struct_member*>(&cpp_member);
}

struct xls_dslx_type_annotation* xls_dslx_struct_member_get_type(
    struct xls_dslx_struct_member* member) {
  auto* cpp_member = reinterpret_cast<xls::dslx::StructMember*>(member);
  xls::dslx::TypeAnnotation* cpp_type_annotation = cpp_member->type;
  return reinterpret_cast<xls_dslx_type_annotation*>(cpp_type_annotation);
}

bool xls_dslx_type_info_get_const_expr(struct xls_dslx_type_info* type_info, struct xls_dslx_expr* expr, char** error_out, struct xls_dslx_interp_value** result_out) {
  auto* cpp_type_info = reinterpret_cast<xls::dslx::TypeInfo*>(type_info);
  auto* cpp_expr = reinterpret_cast<xls::dslx::Expr*>(expr);
  absl::StatusOr<xls::dslx::InterpValue> value_or = cpp_type_info->GetConstExpr(cpp_expr);
  if (!value_or.ok()) {
    *result_out = nullptr;
    *error_out = xls::ToOwnedCString(value_or.status().ToString());
    return false;
  }

  auto* heap = new xls::dslx::InterpValue{std::move(value_or).value()};
  *result_out = reinterpret_cast<xls_dslx_interp_value*>(heap);
  *error_out = nullptr;
  return true;
}

// -- interp_value

void xls_dslx_interp_value_free(struct xls_dslx_interp_value* v) {
  auto* cpp_interp_value = reinterpret_cast<xls::dslx::InterpValue*>(v);
  delete cpp_interp_value;
}

bool xls_dslx_interp_value_convert_to_ir(struct xls_dslx_interp_value* v, char** error_out, struct xls_value** result_out) {
  auto* cpp_interp_value = reinterpret_cast<xls::dslx::InterpValue*>(v);
  absl::StatusOr<xls::Value> ir_value_or = cpp_interp_value->ConvertToIr();
  if (!ir_value_or.ok()) {
    *error_out = xls::ToOwnedCString(ir_value_or.status().ToString());
    *result_out = nullptr;
    return false;
  }

  auto* heap = new xls::Value{std::move(ir_value_or).value()};
  *result_out = reinterpret_cast<xls_value*>(heap);
  *error_out = nullptr;
  return true;
}

}  // extern "C"
