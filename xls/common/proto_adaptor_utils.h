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

#include "absl/strings/string_view.h"

namespace xls {

// Returns a string type with contents equivalent to the string view.
// The generated proto API for strings does not support string_view assignment.
//
// This gives us a convenient way to note which "string view to string"
// conversions are done "just for conformance with the open source API's need
// for full `std::string`s" -- if at some point the protobuf APIs are upgraded
// in open source, all of these conversion calls could be removed.
inline std::string ToProtoString(absl::string_view s) { return std::string(s); }

}  // namespace xls

#endif  // XLS_COMMON_PROTO_ADAPTOR_UTILS_H_
