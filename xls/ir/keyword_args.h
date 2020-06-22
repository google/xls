// Copyright 2020 Google LLC
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

#ifndef XLS_IR_KEYWORD_ARGS_H_
#define XLS_IR_KEYWORD_ARGS_H_

#include <string>
#include <vector>

#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/value.h"

namespace xls {

// Converts the given set of keyword args for the given function into a vector
// of positional arguments.
xabsl::StatusOr<std::vector<Value>> KeywordArgsToPositional(
    const Function& function,
    const absl::flat_hash_map<std::string, Value>& kwargs);

}  // namespace xls

#endif  // XLS_IR_KEYWORD_ARGS_H_
