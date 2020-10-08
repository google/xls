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

#ifndef XLS_COMMON_LOGGING_NULL_GUARD_H_
#define XLS_COMMON_LOGGING_NULL_GUARD_H_

#include <cstddef>

namespace xls {
namespace logging_internal {

// NullGuard exists such that NullGuard<T>::Guard(v) returns v, unless passed a
// nullptr_t, or a null char* or const char*, in which case it returns "(null)".
// This allows streaming NullGuard<T>::Guard(v) to an output stream without
// hitting undefined behavior for null values.
template <typename T>
struct NullGuard {
  static const T& Guard(const T& v) { return v; }
};
template <>
struct NullGuard<char*> {
  static const char* Guard(const char* v) { return v ? v : "(null)"; }
};
template <>
struct NullGuard<const char*> {
  static const char* Guard(const char* v) { return v ? v : "(null)"; }
};
template <>
struct NullGuard<std::nullptr_t> {
  static const char* Guard(const std::nullptr_t&) { return "(null)"; }
};

}  // namespace logging_internal
}  // namespace xls

#endif  // XLS_COMMON_LOGGING_NULL_GUARD_H_
