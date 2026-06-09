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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_FUZZ_TEST_DOMAIN_VALIDATOR_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_FUZZ_TEST_DOMAIN_VALIDATOR_H_

#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

class FuzzTestDomainValidator {
 public:
  FuzzTestDomainValidator(
      const InferenceTable& table,
      std::optional<const ParametricContext*> parametric_context, TypeInfo& ti,
      WarningCollector& warning_collector, const ImportData& import_data,
      const FileTable& file_table)
      : table_(table),
        parametric_context_(parametric_context),
        ti_(ti),
        warning_collector_(warning_collector),
        import_data_(import_data),
        file_table_(file_table) {}

  // Validates that a fuzz test domain is compatible with the corresponding
  // function parameter type. Returns an error if not compatible.
  absl::Status Validate(const Expr* domain, const Type* param_type,
                        std::string_view param_str);

 private:
  absl::StatusOr<InterpValue> Evaluate(const Expr* expr);

  absl::StatusOr<const Expr*> CompleteStructInstance(const StructInstance* node,
                                                     Module& module);
  absl::StatusOr<const Expr*> CompleteExpr(const Expr* expr, Module& module);

  absl::Status RegisterTypes(const Expr* original, const Expr* completed,
                             TypeInfo& ti);

  absl::Status ValidateScalarDomainValue(const InterpValue& value,
                                         const Type* param_type,
                                         const Span& span,
                                         std::string_view param_str);

  absl::Status ValidateTupleDomainValue(const InterpValue& value,
                                        const TupleType* tuple_type,
                                        const Span& span,
                                        std::string_view param_str);

  absl::Status ValidateStructDomainValue(const InterpValue& value,
                                         const StructType* struct_type,
                                         const Span& span,
                                         std::string_view param_str);

  absl::Status ValidateArrayDomainValue(const InterpValue& value,
                                        const ArrayType* array_type,
                                        const Span& span,
                                        std::string_view param_str);

  absl::Status ValidateScalarRangeDomainValue(const InterpValue& value,
                                              const Type* param_type,
                                              const Span& span,
                                              std::string_view param_str);

  absl::Status ValidateScalarElementOfDomainValue(const InterpValue& value,
                                                  const Type* param_type,
                                                  const Span& span,
                                                  std::string_view param_str);

  absl::Status ValidateFuzzTestDomainValue(const InterpValue& value,
                                           const Type* param_type,
                                           const Span& span,
                                           std::string_view param_str);

  const InferenceTable& table_;
  std::optional<const ParametricContext*> parametric_context_;
  TypeInfo& ti_;
  WarningCollector& warning_collector_;
  const ImportData& import_data_;
  const FileTable& file_table_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_FUZZ_TEST_DOMAIN_VALIDATOR_H_
