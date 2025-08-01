// Copyright 2023 The XLS Authors
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

#include "xls/dslx/type_system/typecheck_test_utils.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info_to_proto.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {

absl::StatusOr<TypecheckResult> Typecheck(std::string_view program,
                                          std::string_view module_name,
                                          ImportData* import_data,
                                          bool add_version_attribute) {
  std::unique_ptr<ImportData> owned_import_data;
  if (import_data == nullptr) {
    owned_import_data = CreateImportDataPtrForTest();
    import_data = owned_import_data.get();
  }
  std::string program_with_version_attribute =
      add_version_attribute
          ? absl::StrCat("#![feature(type_inference_v1)]\n\n", program)
          : std::string(program);
  absl::StatusOr<TypecheckedModule> tm = ParseAndTypecheck(
      program_with_version_attribute, absl::StrCat(module_name, ".x"),
      module_name, import_data);

  if (!tm.ok()) {
    UniformContentFilesystem vfs(program_with_version_attribute);
    TryPrintError(tm.status(), import_data->file_table(), vfs);
    return tm.status();
  }
  // Ensure that we can convert all the type information in the unit tests into
  // its protobuf form.
  XLS_RETURN_IF_ERROR(TypeInfoToProto(*tm->type_info).status());

  return TypecheckResult{std::move(owned_import_data), std::move(*tm)};
}

absl::StatusOr<TypecheckResult> TypecheckV2(std::string_view program,
                                            std::string_view module_name,
                                            ImportData* import_data) {
  return Typecheck(absl::StrCat("#![feature(type_inference_v2)]\n\n", program),
                   module_name, import_data,
                   /*add_version_attribute=*/false);
}

}  // namespace xls::dslx
