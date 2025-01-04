// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_FORMAT_TYPE_MISMATCH_H_
#define XLS_DSLX_TYPE_SYSTEM_FORMAT_TYPE_MISMATCH_H_

#include <string>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Returns a string that displays the mismatch between the "lhs" and "rhs" types
// in more detail in an attempt to be helpful to a DSLX user.
//
// This may include lines that highlight specific differences between the types
// where the structure was the same between lhs and rhs; e.g. for
//
//  lhs: (u32, u64)
//  rhs: (u32, s64)
//
// The returned string will highlight that the discrepancy is between the u64 /
// s64 within the tuples.
//
// The returned string should be assumed to be multi-line but not
// newline-terminated.
absl::StatusOr<std::string> FormatTypeMismatch(const Type& lhs, const Type& rhs,
                                               const FileTable& file_table);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_FORMAT_TYPE_MISMATCH_H_
