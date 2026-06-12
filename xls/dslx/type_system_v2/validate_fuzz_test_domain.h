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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_VALIDATE_FUZZ_TEST_DOMAIN_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_VALIDATE_FUZZ_TEST_DOMAIN_H_

#include <string_view>

#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Validates that a fuzz test domain is compatible with the corresponding
// function parameter type. Returns an error if not compatible.
absl::Status ValidateFuzzTestDomain(const Expr* domain, const Type* param_type,
                                    std::string_view param_str,
                                    const InferenceTable& table,
                                    const TypeInfo& ti,
                                    WarningCollector& warning_collector,
                                    const ImportData& import_data,
                                    const FileTable& file_table);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_VALIDATE_FUZZ_TEST_DOMAIN_H_
