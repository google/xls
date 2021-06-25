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

#ifndef XLS_COMMON_STRING_INTERPOLATION_H_
#define XLS_COMMON_STRING_INTERPOLATION_H_

#include "absl/status/statusor.h"

namespace xls {

// Callback provided by the user for determining how to print individual
// arguments. It is passed the format spec (the text between braces `{}`), as
// well as the index of the argument in question (see InterpolateArgs below).
using InterpolationCallback =
    absl::FunctionRef<absl::StatusOr<std::string>(absl::string_view, int64_t)>;

// Parses the input string, calling the provided callback on any format
// arguments and interpolating them into the original string.
//
// For example, passing a string of "start {} mid {foo} end" will call the
// callback with arguments ("", 0) and ("foo", 1), substituting the call results
// (assuming no failures) into the resulting string. Using javascript syntax,
// this would be equivalent to the template literal:
//
//   `start ${print_arg("", 0)} mid ${print_arg("foo", 1)} end`
//
// This function allows escaping by doubling up on curly braces. For example,
// an input string of "{{}}" will result in an output string of "{}", similar
// to Python and Rust.
absl::StatusOr<std::string> InterpolateArgs(absl::string_view s,
                                            InterpolationCallback print_arg);

}  // namespace xls

#endif  // XLS_COMMON_STRING_INTERPOLATION_H_
