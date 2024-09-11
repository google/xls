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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPECHECK_TEST_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPECHECK_TEST_UTILS_H_

#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {

struct TypecheckResult {
  // If `import_data` is not dependency injected into the `Typecheck` routine,
  // we create import data, and it owns objects with lifetimes we need for the
  // `TypecheckedModule` (e.g. the `FileTable`) so we provide it in the result.
  std::unique_ptr<ImportData> import_data;
  TypecheckedModule tm;
};

// Helper for parsing/typechecking a snippet of DSLX text.
//
// If `import_data` is not provided one is created for internal use.
absl::StatusOr<TypecheckResult> Typecheck(std::string_view text);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPECHECK_TEST_UTILS_H_
