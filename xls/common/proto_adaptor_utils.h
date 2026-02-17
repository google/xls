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

// Utilities for interfacing with proto objects.

#ifndef XLS_COMMON_PROTO_ADAPTOR_UTILS_H_
#define XLS_COMMON_PROTO_ADAPTOR_UTILS_H_

#include <string_view>

#include "absl/base/macros.h"

namespace xls {

// Previously, this function marked places where we had to construct a
// std::string to conform to an API that demanded a string, but we anticipated
// would eventually change to accept string_view. That day has come.
ABSL_DEPRECATE_AND_INLINE()
inline std::string_view ToProtoString(std::string_view s) { return s; }

}  // namespace xls

#endif  // XLS_COMMON_PROTO_ADAPTOR_UTILS_H_
