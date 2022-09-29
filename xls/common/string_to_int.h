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

#ifndef XLS_COMMON_STRING_TO_INT_H_
#define XLS_COMMON_STRING_TO_INT_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xls {

// Converts s into an int64_t using the given base.
//
// Note: if a base is provided, the "leader" that indicates the base is
// attempted to be consumed at the front of "s"; e.g. if you provide base 2
// we'll attempt to prefix-consume a "0b".
//
// Warning: be careful if you provide a base of 0 and have leading zeros --
// octal base will be inferred from the first leading zero!
//
// InvalidArgumentError: if invalid digits are present in the string for the
// determined base, or if the value overflows a 64-bit value.
absl::StatusOr<int64_t> StrTo64Base(std::string_view s, int base);

}  // namespace xls

#endif  // XLS_COMMON_STRING_TO_INT_H_
