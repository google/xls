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

#ifndef XLS_DSLX_IR_CONVERT_FUZZ_TEST_CONVERTER_H_
#define XLS_DSLX_IR_CONVERT_FUZZ_TEST_CONVERTER_H_

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/attribute_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {

class FuzzTestConverter {
 public:
  FuzzTestConverter(TypeInfo* type_info, ImportData* import_data)
      : current_type_info_(type_info), import_data_(import_data) {}

  absl::StatusOr<std::optional<AttributeData>> LowerFuzzTestDomains(
      const Function* node);

 private:
  absl::Status LowerConstant(const Type* param_type, const InterpValue& val,
                             PackageInterfaceProto::FuzzTestDomain& proto);
  // Lower an enum type as an ElementOf domain containing each of the enum
  // values.
  absl::Status LowerArbitraryEnum(const Type* param_type,
                                  PackageInterfaceProto::FuzzTestDomain& proto);
  absl::Status LowerArbitraryType(const Type* param_type,
                                  PackageInterfaceProto::FuzzTestDomain& proto);
  absl::Status LowerTuple(const Type* param_type,
                          const std::vector<InterpValue>& elements,
                          PackageInterfaceProto::FuzzTestDomain& proto);

  absl::Status LowerStructInstanceDomain(
      const StructType& struct_type, const StructInstance& struct_domain,
      PackageInterfaceProto::FuzzTestDomain& proto);

  absl::Status LowerRangeExpr(const Range* range_node, TypeInfo* type_info,
                              PackageInterfaceProto::FuzzTestDomain& proto);
  // Main entry point
  absl::Status LowerDomainExpr(const Type* param_type, const Expr* expr,
                               PackageInterfaceProto::FuzzTestDomain& proto);

 private:
  TypeInfo* current_type_info_;
  ImportData* import_data_;
};
}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_FUZZTEST_CONVERTER_H_
