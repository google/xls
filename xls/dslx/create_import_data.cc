// Copyright 2022 The XLS Authors
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
#include "xls/dslx/create_import_data.h"

#include <memory>

#include "xls/dslx/default_dslx_stdlib_path.h"

namespace xls::dslx {

ImportData CreateImportData(
    std::string stdlib_path,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  ImportData import_data(stdlib_path, additional_search_paths);
  import_data.SetBytecodeCache(std::make_unique<BytecodeCache>(&import_data));
  return import_data;
}

ImportData CreateImportDataForTest() {
  ImportData import_data(xls::kDefaultDslxStdlibPath,
                         /*additional_search_paths=*/{});
  import_data.SetBytecodeCache(std::make_unique<BytecodeCache>(&import_data));
  return import_data;
}

std::unique_ptr<ImportData> CreateImportDataPtrForTest() {
  auto import_data =
      absl::WrapUnique(new ImportData(xls::kDefaultDslxStdlibPath,
                                      /*additional_search_paths=*/{}));
  import_data->SetBytecodeCache(
      std::make_unique<BytecodeCache>(import_data.get()));
  return import_data;
}

}  // namespace xls::dslx
