// Copyright 2020 The XLS Authors
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

// Define strong_ints used by source locators.

#ifndef XLS_IR_FILENO_H_
#define XLS_IR_FILENO_H_

#include "absl/strings/string_view.h"
#include "xls/common/integral_types.h"
#include "xls/common/strong_int.h"

namespace xls {

DEFINE_STRONG_INT_TYPE(Fileno, int32);
DEFINE_STRONG_INT_TYPE(Lineno, int32);
DEFINE_STRONG_INT_TYPE(Colno, int32);

}  // namespace xls

#endif  // XLS_IR_FILENO_H_
