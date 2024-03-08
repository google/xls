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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPE_ZERO_VALUE_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPE_ZERO_VALUE_H_

#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Make a zero-value of this concrete type.
//
// This is not always possible, e.g. when there is an enum that does not have a
// defined zero value, in which cases an error is returned. In this case, span
// is used to cite the source of the error in the program text.
absl::StatusOr<InterpValue> MakeZeroValue(const Type& type,
                                          const ImportData& import_data,
                                          const Span& span);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPE_ZERO_VALUE_H_
