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
#ifndef XLS_DSLX_CREATE_IMPORT_DATA_H_
#define XLS_DSLX_CREATE_IMPORT_DATA_H_

// Routines for creating and initializing ImportData objects. Due to dependency
// issues between InterpData and BytecodeEmitter, we need an interstitial
// library here.

#include <filesystem>  // NOLINT
#include <memory>

#include "absl/types/span.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

// Creates an ImportData with the given stdlib and search paths and assigns a
// BytecodeCache as the bytecode cache on the result.
ImportData CreateImportData(
    const std::filesystem::path& stdlib_path,
    absl::Span<const std::filesystem::path> additional_search_paths,
    WarningKindSet warnings, std::unique_ptr<VirtualizableFilesystem> vfs);

// Creates an ImportData with reasonable defaults (standard path to the stdlib
// and no additional search paths).
ImportData CreateImportDataForTest(
    std::unique_ptr<VirtualizableFilesystem> vfs = nullptr,
    WarningKindSet warnings = kAllWarningsSet);

std::unique_ptr<ImportData> CreateImportDataPtrForTest();

}  // namespace xls::dslx

#endif  // XLS_DSLX_CREATE_IMPORT_DATA_H_
