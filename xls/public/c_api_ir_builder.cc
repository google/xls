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

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
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

struct xls_type* xls_package_get_array_type(struct xls_package* package,
                                            struct xls_type* element_type,
                                            int64_t size) {
  auto* cpp_package = reinterpret_cast<xls::Package*>(package);
  auto* cpp_element_type = reinterpret_cast<xls::Type*>(element_type);
  auto* cpp_type = cpp_package->GetArrayType(size, cpp_element_type);
  return reinterpret_cast<xls_type*>(cpp_type);
}

struct xls_type* xls_package_get_tuple_type(struct xls_package* package,
                                            struct xls_type** members,
                                            int64_t member_count) {
  auto* cpp_package = reinterpret_cast<xls::Package*>(package);
  std::vector<xls::Type*> cpp_members;
  cpp_members.reserve(member_count);
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

static xls_bvalue* xls_builder_base_unop_generic(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    const char* name,
    std::function<xls::BValue(xls::BuilderBase*, xls::BValue, xls::SourceInfo,
                              std::string_view)>
        f) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_value = reinterpret_cast<xls::BValue*>(value);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = f(cpp_builder, *cpp_value, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

static xls_bvalue* xls_builder_base_binop_generic(
    struct xls_builder_base* builder, struct xls_bvalue* lhs,
    struct xls_bvalue* rhs, const char* name,
    std::function<xls::BValue(xls::BuilderBase*, xls::BValue, xls::BValue,
                              xls::SourceInfo, std::string_view)>
        f) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_lhs = reinterpret_cast<xls::BValue*>(lhs);
  auto* cpp_rhs = reinterpret_cast<xls::BValue*>(rhs);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      f(cpp_builder, *cpp_lhs, *cpp_rhs, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

static xls_bvalue* xls_builder_base_binop(struct xls_builder_base* builder,
                                          xls::Op op, struct xls_bvalue* lhs,
                                          struct xls_bvalue* rhs,
                                          const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [op](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
           xls::SourceInfo loc, std::string_view name) {
        return builder->AddBinOp(op, lhs, rhs, loc, name);
      });
}

struct xls_bvalue* xls_builder_base_add_and(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->And(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_nand(struct xls_builder_base* builder,
                                             struct xls_bvalue* lhs,
                                             struct xls_bvalue* rhs,
                                             const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->Nand(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_or(struct xls_builder_base* builder,
                                           struct xls_bvalue* lhs,
                                           struct xls_bvalue* rhs,
                                           const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->Or(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_not(struct xls_builder_base* builder,
                                            struct xls_bvalue* value,
                                            const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [](xls::BuilderBase* builder, xls::BValue value, xls::SourceInfo loc,
         std::string_view name) { return builder->Not(value, loc, name); });
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
  cpp_operands.reserve(operand_count);
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

struct xls_bvalue* xls_builder_base_add_array(struct xls_builder_base* builder,
                                              struct xls_type* element_type,
                                              struct xls_bvalue** elements,
                                              int64_t element_count,
                                              const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_element_type = reinterpret_cast<xls::Type*>(element_type);
  std::vector<xls::BValue> cpp_elements;
  cpp_elements.reserve(element_count);
  for (int64_t i = 0; i < element_count; ++i) {
    cpp_elements.push_back(*reinterpret_cast<xls::BValue*>(elements[i]));
  }
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      cpp_builder->Array(absl::MakeConstSpan(cpp_elements), cpp_element_type,
                         xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_array_index(
    struct xls_builder_base* builder, struct xls_bvalue* array,
    struct xls_bvalue** indices, int64_t index_count, bool assumed_in_bounds,
    const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_array = reinterpret_cast<xls::BValue*>(array);
  std::vector<xls::BValue> cpp_indices;
  cpp_indices.reserve(index_count);
  for (int64_t i = 0; i < index_count; ++i) {
    cpp_indices.push_back(*reinterpret_cast<xls::BValue*>(indices[i]));
  }
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = cpp_builder->ArrayIndex(
      *cpp_array, cpp_indices, assumed_in_bounds, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_bit_slice(
    struct xls_builder_base* builder, struct xls_bvalue* value, int64_t start,
    int64_t width, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_value = reinterpret_cast<xls::BValue*>(value);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = cpp_builder->BitSlice(*cpp_value, start, width,
                                             xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_bit_slice_update(
    struct xls_builder_base* builder, struct xls_bvalue* arg,
    struct xls_bvalue* start, struct xls_bvalue* update_value,
    const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_arg = reinterpret_cast<xls::BValue*>(arg);
  auto* cpp_start = reinterpret_cast<xls::BValue*>(start);
  auto* cpp_update_value = reinterpret_cast<xls::BValue*>(update_value);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = cpp_builder->BitSliceUpdate(
      *cpp_arg, *cpp_start, *cpp_update_value, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_dynamic_bit_slice(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    struct xls_bvalue* start, int64_t width, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_value = reinterpret_cast<xls::BValue*>(value);
  auto* cpp_start = reinterpret_cast<xls::BValue*>(start);
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = cpp_builder->DynamicBitSlice(
      *cpp_value, *cpp_start, width, xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_concat(struct xls_builder_base* builder,
                                               struct xls_bvalue** operands,
                                               int64_t operand_count,
                                               const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  std::vector<xls::BValue> cpp_operands;
  cpp_operands.reserve(operand_count);
  for (int64_t i = 0; i < operand_count; ++i) {
    cpp_operands.push_back(*reinterpret_cast<xls::BValue*>(operands[i]));
  }
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = cpp_builder->Concat(absl::MakeConstSpan(cpp_operands),
                                           xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_shra(struct xls_builder_base* builder,
                                             struct xls_bvalue* value,
                                             struct xls_bvalue* amount,
                                             const char* name) {
  return xls_builder_base_binop(builder, xls::Op::kShra, value, amount, name);
}

struct xls_bvalue* xls_builder_base_add_shrl(struct xls_builder_base* builder,
                                             struct xls_bvalue* value,
                                             struct xls_bvalue* amount,
                                             const char* name) {
  return xls_builder_base_binop(builder, xls::Op::kShrl, value, amount, name);
}

struct xls_bvalue* xls_builder_base_add_shll(struct xls_builder_base* builder,
                                             struct xls_bvalue* value,
                                             struct xls_bvalue* amount,
                                             const char* name) {
  return xls_builder_base_binop(builder, xls::Op::kShll, value, amount, name);
}

struct xls_bvalue* xls_builder_base_add_nor(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->Nor(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_add(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop(builder, xls::Op::kAdd, lhs, rhs, name);
}

struct xls_bvalue* xls_builder_base_add_sub(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop(builder, xls::Op::kSub, lhs, rhs, name);
}

struct xls_bvalue* xls_builder_base_add_umul(struct xls_builder_base* builder,
                                             struct xls_bvalue* lhs,
                                             struct xls_bvalue* rhs,
                                             const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->UMul(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_smul(struct xls_builder_base* builder,
                                             struct xls_bvalue* lhs,
                                             struct xls_bvalue* rhs,
                                             const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->SMul(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_umulp(struct xls_builder_base* builder,
                                              struct xls_bvalue* lhs,
                                              struct xls_bvalue* rhs,
                                              const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* cpp_builder, xls::BValue cpp_lhs,
         xls::BValue cpp_rhs, xls::SourceInfo loc, std::string_view cpp_name) {
        return cpp_builder->UMulp(cpp_lhs, cpp_rhs, loc, cpp_name);
      });
}

struct xls_bvalue* xls_builder_base_add_smulp(struct xls_builder_base* builder,
                                              struct xls_bvalue* lhs,
                                              struct xls_bvalue* rhs,
                                              const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* cpp_builder, xls::BValue cpp_lhs,
         xls::BValue cpp_rhs, xls::SourceInfo loc, std::string_view cpp_name) {
        return cpp_builder->SMulp(cpp_lhs, cpp_rhs, loc, cpp_name);
      });
}

struct xls_bvalue* xls_builder_base_add_udiv(struct xls_builder_base* builder,
                                             struct xls_bvalue* lhs,
                                             struct xls_bvalue* rhs,
                                             const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* cpp_builder, xls::BValue cpp_lhs,
         xls::BValue cpp_rhs, xls::SourceInfo loc, std::string_view cpp_name) {
        return cpp_builder->UDiv(cpp_lhs, cpp_rhs, loc, cpp_name);
      });
}

struct xls_bvalue* xls_builder_base_add_sdiv(struct xls_builder_base* builder,
                                             struct xls_bvalue* lhs,
                                             struct xls_bvalue* rhs,
                                             const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* cpp_builder, xls::BValue cpp_lhs,
         xls::BValue cpp_rhs, xls::SourceInfo loc, std::string_view cpp_name) {
        return cpp_builder->SDiv(cpp_lhs, cpp_rhs, loc, cpp_name);
      });
}

struct xls_bvalue* xls_builder_base_add_umod(struct xls_builder_base* builder,
                                             struct xls_bvalue* lhs,
                                             struct xls_bvalue* rhs,
                                             const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* cpp_builder, xls::BValue cpp_lhs,
         xls::BValue cpp_rhs, xls::SourceInfo loc, std::string_view cpp_name) {
        return cpp_builder->UMod(cpp_lhs, cpp_rhs, loc, cpp_name);
      });
}

struct xls_bvalue* xls_builder_base_add_smod(struct xls_builder_base* builder,
                                             struct xls_bvalue* lhs,
                                             struct xls_bvalue* rhs,
                                             const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* cpp_builder, xls::BValue cpp_lhs,
         xls::BValue cpp_rhs, xls::SourceInfo loc, std::string_view cpp_name) {
        return cpp_builder->SMod(cpp_lhs, cpp_rhs, loc, cpp_name);
      });
}

struct xls_bvalue* xls_builder_base_add_xor(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->Xor(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_one_hot_select(
    struct xls_builder_base* builder, struct xls_bvalue* selector,
    struct xls_bvalue** cases, int64_t case_count, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_selector = reinterpret_cast<xls::BValue*>(selector);
  std::vector<xls::BValue> cpp_cases;
  cpp_cases.reserve(case_count);
  for (int64_t i = 0; i < case_count; ++i) {
    cpp_cases.push_back(*reinterpret_cast<xls::BValue*>(cases[i]));
  }
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue = cpp_builder->OneHotSelect(*cpp_selector, cpp_cases,
                                                 xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_priority_select(
    struct xls_builder_base* builder, struct xls_bvalue* selector,
    struct xls_bvalue** cases, int64_t case_count,
    struct xls_bvalue* default_value, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_selector = reinterpret_cast<xls::BValue*>(selector);
  auto* cpp_default_value = reinterpret_cast<xls::BValue*>(default_value);
  std::vector<xls::BValue> cpp_cases;
  cpp_cases.reserve(case_count);
  for (int64_t i = 0; i < case_count; ++i) {
    cpp_cases.push_back(*reinterpret_cast<xls::BValue*>(cases[i]));
  }
  std::string_view cpp_name = name == nullptr ? "" : name;
  xls::BValue bvalue =
      cpp_builder->PrioritySelect(*cpp_selector, cpp_cases, *cpp_default_value,
                                  xls::SourceInfo(), cpp_name);
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_eq(struct xls_builder_base* builder,
                                           struct xls_bvalue* lhs,
                                           struct xls_bvalue* rhs,
                                           const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->Eq(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_ne(struct xls_builder_base* builder,
                                           struct xls_bvalue* lhs,
                                           struct xls_bvalue* rhs,
                                           const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->Ne(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_ult(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->ULt(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_ule(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->ULe(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_ugt(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->UGt(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_uge(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->UGe(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_slt(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->SLt(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_sle(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->SLe(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_sgt(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->SGt(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_sge(struct xls_builder_base* builder,
                                            struct xls_bvalue* lhs,
                                            struct xls_bvalue* rhs,
                                            const char* name) {
  return xls_builder_base_binop_generic(
      builder, lhs, rhs, name,
      [](xls::BuilderBase* builder, xls::BValue lhs, xls::BValue rhs,
         xls::SourceInfo loc,
         std::string_view name) { return builder->SGe(lhs, rhs, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_and_reduce(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [](xls::BuilderBase* builder, xls::BValue value, xls::SourceInfo loc,
         std::string_view name) {
        return builder->AndReduce(value, loc, name);
      });
}

struct xls_bvalue* xls_builder_base_add_or_reduce(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [](xls::BuilderBase* builder, xls::BValue value, xls::SourceInfo loc,
         std::string_view name) {
        return builder->OrReduce(value, loc, name);
      });
}

struct xls_bvalue* xls_builder_base_add_xor_reduce(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [](xls::BuilderBase* builder, xls::BValue value, xls::SourceInfo loc,
         std::string_view name) {
        return builder->XorReduce(value, loc, name);
      });
}

struct xls_bvalue* xls_builder_base_add_negate(struct xls_builder_base* builder,
                                               struct xls_bvalue* value,
                                               const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [](xls::BuilderBase* builder, xls::BValue value, xls::SourceInfo loc,
         std::string_view name) { return builder->Negate(value, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_reverse(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [](xls::BuilderBase* builder, xls::BValue value, xls::SourceInfo loc,
         std::string_view name) { return builder->Reverse(value, loc, name); });
}

struct xls_bvalue* xls_builder_base_add_one_hot(
    struct xls_builder_base* builder, struct xls_bvalue* input,
    bool lsb_is_priority, const char* name) {
  return xls_builder_base_unop_generic(
      builder, input, name,
      [lsb_is_priority](xls::BuilderBase* builder, xls::BValue value,
                        xls::SourceInfo loc, std::string_view name) {
        return builder->OneHot(
            value, lsb_is_priority ? xls::LsbOrMsb::kLsb : xls::LsbOrMsb::kMsb,
            loc, name);
      });
}

struct xls_bvalue* xls_builder_base_add_sign_extend(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    int64_t new_bit_count, const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [new_bit_count](xls::BuilderBase* builder, xls::BValue value,
                      xls::SourceInfo loc, std::string_view name) {
        return builder->SignExtend(value, new_bit_count, loc, name);
      });
}

struct xls_bvalue* xls_builder_base_add_zero_extend(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    int64_t new_bit_count, const char* name) {
  return xls_builder_base_unop_generic(
      builder, value, name,
      [new_bit_count](xls::BuilderBase* builder, xls::BValue value,
                      xls::SourceInfo loc, std::string_view name) {
        return builder->ZeroExtend(value, new_bit_count, loc, name);
      });
}

xls_type* xls_builder_base_get_type(struct xls_builder_base* builder,
                                    struct xls_bvalue* value) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  auto* cpp_value = reinterpret_cast<xls::BValue*>(value);
  xls::Type* cpp_type = cpp_builder->GetType(*cpp_value);
  return reinterpret_cast<xls_type*>(cpp_type);
}

bool xls_builder_base_get_last_value(struct xls_builder_base* builder,
                                     char** error_out,
                                     struct xls_bvalue** value_out) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);

  if (absl::Status error = cpp_builder->GetError(); !error.ok()) {
    *error_out = xls::ToOwnedCString(error.ToString());
    return false;
  }

  absl::StatusOr<xls::BValue> cpp_last_value = cpp_builder->GetLastValue();
  if (!cpp_last_value.ok()) {
    *error_out = xls::ToOwnedCString(cpp_last_value.status().ToString());
    return false;
  }
  auto* cpp_bvalue = new xls::BValue(*cpp_last_value);
  *value_out = reinterpret_cast<xls_bvalue*>(cpp_bvalue);
  return true;
}

struct xls_type* xls_package_get_token_type(struct xls_package* package) {
  auto* cpp_package = reinterpret_cast<xls::Package*>(package);
  auto* cpp_type = cpp_package->GetTokenType();
  return reinterpret_cast<xls_type*>(cpp_type);
}

struct xls_bvalue* xls_builder_base_add_select(struct xls_builder_base* builder,
                                               struct xls_bvalue* selector,
                                               struct xls_bvalue** cases,
                                               int64_t case_count,
                                               struct xls_bvalue* default_value,
                                               const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_selector = *reinterpret_cast<xls::BValue*>(selector);
  std::vector<xls::BValue> cpp_cases;
  cpp_cases.reserve(case_count);
  for (int64_t i = 0; i < case_count; ++i) {
    cpp_cases.push_back(*reinterpret_cast<xls::BValue*>(cases[i]));
  }
  std::optional<xls::BValue> cpp_default = std::nullopt;
  if (default_value != nullptr) {
    cpp_default = *reinterpret_cast<xls::BValue*>(default_value);
  }
  xls::BValue bvalue = cpp_builder->Select(cpp_selector, cpp_cases, cpp_default,
                                           xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_clz(struct xls_builder_base* builder,
                                            struct xls_bvalue* value,
                                            const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_value = *reinterpret_cast<xls::BValue*>(value);
  xls::BValue bvalue =
      cpp_builder->Clz(cpp_value, xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_ctz(struct xls_builder_base* builder,
                                            struct xls_bvalue* value,
                                            const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_value = *reinterpret_cast<xls::BValue*>(value);
  xls::BValue bvalue =
      cpp_builder->Ctz(cpp_value, xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_after_all(
    struct xls_builder_base* builder, struct xls_bvalue** dependencies,
    int64_t dependency_count, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  std::vector<xls::BValue> cpp_deps;
  cpp_deps.reserve(dependency_count);
  for (int64_t i = 0; i < dependency_count; ++i) {
    cpp_deps.push_back(*reinterpret_cast<xls::BValue*>(dependencies[i]));
  }
  xls::BValue bvalue =
      cpp_builder->AfterAll(cpp_deps, xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_array_slice(
    struct xls_builder_base* builder, struct xls_bvalue* array,
    struct xls_bvalue* start, int64_t width, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_array = *reinterpret_cast<xls::BValue*>(array);
  xls::BValue cpp_start = *reinterpret_cast<xls::BValue*>(start);
  xls::BValue bvalue = cpp_builder->ArraySlice(
      cpp_array, cpp_start, width, xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_array_update(
    struct xls_builder_base* builder, struct xls_bvalue* array,
    struct xls_bvalue* update_value, struct xls_bvalue** indices,
    int64_t index_count, bool assumed_in_bounds, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_array = *reinterpret_cast<xls::BValue*>(array);
  xls::BValue cpp_update = *reinterpret_cast<xls::BValue*>(update_value);
  std::vector<xls::BValue> cpp_indices;
  cpp_indices.reserve(index_count);
  for (int64_t i = 0; i < index_count; ++i) {
    cpp_indices.push_back(*reinterpret_cast<xls::BValue*>(indices[i]));
  }
  xls::BValue bvalue = cpp_builder->ArrayUpdate(
      cpp_array, cpp_update, cpp_indices, assumed_in_bounds, xls::SourceInfo(),
      name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_array_concat(
    struct xls_builder_base* builder, struct xls_bvalue** arrays,
    int64_t array_count, const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  std::vector<xls::BValue> cpp_arrays;
  cpp_arrays.reserve(array_count);
  for (int64_t i = 0; i < array_count; ++i) {
    cpp_arrays.push_back(*reinterpret_cast<xls::BValue*>(arrays[i]));
  }
  xls::BValue bvalue =
      cpp_builder->ArrayConcat(cpp_arrays, xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_identity(
    struct xls_builder_base* builder, struct xls_bvalue* value,
    const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_value = *reinterpret_cast<xls::BValue*>(value);
  xls::BValue bvalue =
      cpp_builder->Identity(cpp_value, xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_encode(struct xls_builder_base* builder,
                                               struct xls_bvalue* value,
                                               const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_value = *reinterpret_cast<xls::BValue*>(value);
  xls::BValue bvalue =
      cpp_builder->Encode(cpp_value, xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

struct xls_bvalue* xls_builder_base_add_decode(struct xls_builder_base* builder,
                                               struct xls_bvalue* value,
                                               int64_t* width,
                                               const char* name) {
  auto* cpp_builder = reinterpret_cast<xls::BuilderBase*>(builder);
  xls::BValue cpp_value = *reinterpret_cast<xls::BValue*>(value);
  std::optional<int64_t> cpp_width = std::nullopt;
  if (width != nullptr) {
    cpp_width = *width;
  }
  xls::BValue bvalue = cpp_builder->Decode(cpp_value, cpp_width,
                                           xls::SourceInfo(), name ? name : "");
  auto* cpp_heap_bvalue = new xls::BValue(bvalue);
  return reinterpret_cast<xls_bvalue*>(cpp_heap_bvalue);
}

}  // extern "C"
