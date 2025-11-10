// Copyright 2021 The XLS Authors
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

#ifndef XLS_PUBLIC_RUNTIME_IR_OPT_ACTIONS_H_
#define XLS_PUBLIC_RUNTIME_IR_OPT_ACTIONS_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"

namespace xls {

absl::StatusOr<std::string> OptimizeIr(std::string_view ir,
                                       std::string_view top);

}  // namespace xls

#endif  // XLS_PUBLIC_RUNTIME_IR_OPT_ACTIONS_H_
