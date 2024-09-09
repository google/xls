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

#ifndef XLS_DSLX_TYPE_SYSTEM_MAYBE_EXPLAIN_ERROR_H_
#define XLS_DSLX_TYPE_SYSTEM_MAYBE_EXPLAIN_ERROR_H_

#include "absl/status/status.h"
#include "xls/dslx/type_system/type_mismatch_error_data.h"

namespace xls::dslx {

// Attempts to give a diagnostic explanation for a type mismatch error, if it is
// positional and there are additional diagnostics for it. Otherwise, passes the
// original error data back as an absl::Status form similar to other DSLX
// errors.
absl::Status MaybeExplainError(const TypeMismatchErrorData& data,
                               const FileTable& file_table);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_MAYBE_EXPLAIN_ERROR_H_
