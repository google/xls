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
//
// API for capturing source-code location information.
// Based on http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4519.pdf.
//
// To define a function that has access to the source location of the
// callsite, define it with a parameter of type `xabsl::SourceLocation`. The
// caller can then invoke the function, passing `XABSL_LOC` as the argument.
//
// If at all possible, make the `xabsl::SourceLocation` parameter be the
// function's last parameter. That way, when `std::source_location` is
// available, you will be able to switch to it, and give the parameter a default
// argument of `std::source_location::current()`. Users will then be able to
// omit that argument, and the default will automatically capture the location
// of the callsite.

#ifndef XLS_COMMON_SOURCE_LOCATION_H_
#define XLS_COMMON_SOURCE_LOCATION_H_

#include "absl/types/source_location.h"

// Use absl::SourceLocation
namespace xabsl {
using SourceLocation = ::absl::SourceLocation;
#define XABSL_LOC ::absl::SourceLocation::current()
#define XABSL_LOC_CURRENT_DEFAULT_ARG = ::absl::SourceLocation::current()
}  // namespace xabsl

#endif  // XLS_COMMON_SOURCE_LOCATION_H_
