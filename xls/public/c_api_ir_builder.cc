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

#include "xls/public/c_api_ir_builder.h"

#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/value.h"
#include "xls/public/c_api_impl_helpers.h"

extern "C" {

struct xls_package* xls_package_create(const char* name) {
  auto* cpp_package = new xls::Package(name);
  return reinterpret_cast<xls_package*>(cpp_package);
}

struct xls_type* xls_package_get_bits_type(struct xls_package* package,
                                           int64_t bit_count) {
  auto* cpp_package = reinterpret_cast<xls::Package*>(package);
  auto* cpp_type = cpp_package->GetBitsType(bit_count);
  return reinterpret_cast<xls_type*>(cpp_type);
}

struct xls_type* xls_package_get_tuple_type(struct xls_package* package,
                                            struct xls_type** members,
                                            int64_t member_count) {
  auto* cpp_package = reinterpret_cast<xls::Package*>(package);
  std::vector<xls::Type*> cpp_members;
  for (int64_t i = 0; i < member_count; ++i) {
    cpp_members.push_back(reinterpret_cast<xls::Type*>(members[i]));
  }
  auto* cpp_type = cpp_package->GetTupleType(absl::MakeConstSpan(cpp_members));
  return reinterpret_cast<xls_type*>(cpp_type);
}

struct xls_function_builder* xls_function_builder_create(
    const char* name, struct xls_package* package, bool should_verify) {
  auto* cpp_package = reinterpret_cast<xls::Package*>(package);
  auto* cpp_builder =
      new xls::FunctionBuilder(name, cpp_package, should_verify);
  return reinterpret_cast<xls_function_builder*>(cpp_builder);
}

void xls_function_builder_free(struct xls_function_builder* builder) {
  delete reinterpret_cast<xls::FunctionBuilder*>(builder);
}

void xls_bvalue_free(struct xls_bvalue* bvalue) {
  delete reinterpret_cast<xls::BValue*>(bvalue);
}

struct xls_builder_base* xls_function_builder_as_builder_base(
    struct xls_function_builder* fn_builder) {
  return reinterpret_cast<xls_builder_base*>(fn_builder);
}

struct xls_bvalue* xls_function_builder_add_parameter(
    struct xls_function_builder* builder, const char* name,
    struct xls_type* type) {
  auto* cpp_builder = reinterpret_cast<xls::FunctionBuilder*>(builder);
  auto* cpp_type = reinterpret_cast<xls::Type*>(type);
  xls::BValue bvalue = cpp_builder->Param(name, cpp_type);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

bool xls_function_builder_build(struct xls_function_builder* builder,
                                char** error_out,
                                struct xls_function** function_out) {
  auto* cpp_builder = reinterpret_cast<xls::FunctionBuilder*>(builder);
  absl::StatusOr<xls::Function*> cpp_function = cpp_builder->Build();
  if (!cpp_function.ok()) {
    *error_out = xls::ToOwnedCString(cpp_function.status().ToString());
    return false;
  }
  *function_out = reinterpret_cast<xls_function*>(cpp_function.value());
  return true;
}

bool xls_function_builder_build_with_return_value(
    struct xls_function_builder* builder, struct xls_bvalue* return_value,
    char** error_out, struct xls_function** function_out) {
  auto* cpp_builder = reinterpret_cast<xls::FunctionBuilder*>(builder);
  auto* cpp_return_value = reinterpret_cast<xls::BValue*>(return_value);
  absl::StatusOr<xls::Function*> cpp_function =
      cpp_builder->BuildWithReturnValue(*cpp_return_value);
  if (!cpp_function.ok()) {
    *error_out = xls::ToOwnedCString(cpp_function.status().ToString());
    return false;
  }
  *function_out = reinterpret_cast<xls_function*>(cpp_function.value());
  return true;
}

// -- xls_builder_base

struct xls_bvalue* xls_builder_base_add_and(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_lhs = reinterpret_cast<xls::BValue*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::BValue*>(rhs);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      cpp_builder->And(*cpp_lhs, *cpp_rhs, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_or(struct xls_builder_base* builder,
                                           struct xls_bvalue* lhs,
                                           struct xls_bvalue* rhs,
                                           const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_lhs = reinterpret_cast<xls::BValue*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::BValue*>(rhs);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      cpp_builder->Or(*cpp_lhs, *cpp_rhs, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_not(struct xls_builder_base* builder,
                                            struct xls_bvalue* value,
                                            const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_value = reinterpret_cast<xls::BValue*>(value);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      cpp_builder->Not(*cpp_value, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_literal(
    struct xls_builder_base* builder, struct xls_value* value,
    const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_value = reinterpret_cast<xls::Value*>(value);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      cpp_builder->Literal(*cpp_value, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_tuple(struct xls_builder_base* builder,
                                              struct xls_bvalue** operands,
                                              int64_t operand_count,
                                              const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  std::vector<xls::BValue> cpp_operands;
  for (int64_t i = 0; i < operand_count; ++i) {
    cpp_operands.push_back(*reinterpret_cast<xls::BValue*>(operands[i]));
  }
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = cpp_builder->Tuple(absl::MakeConstSpan(cpp_operands),
                                          xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_tuple_index(
    struct xls_builder_base* builder, struct xls_bvalue* tuple, int64_t index,
    const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_tuple = reinterpret_cast<xls::BValue*>(tuple);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      cpp_builder->TupleIndex(*cpp_tuple, index, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

}  // extern "C"
