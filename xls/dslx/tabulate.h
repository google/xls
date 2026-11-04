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

#ifndef XLS_DSLX_TABULATE_H_
#define XLS_DSLX_TABULATE_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {

// Looks up `fn_name` in `tm.module`, evaluates the non-parametric function
// annotated with `#[tabulate]` across its input domain, and renders the
// resulting Markdown table.
absl::StatusOr<std::string> TabulateFunction(ImportData* import_data,
                                             const TypecheckedModule& tm,
                                             std::string_view fn_name);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TABULATE_H_
