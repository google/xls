// Copyright 2022 The XLS Authors
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

#ifndef XLS_COMMON_TO_STRING_HELPERS_H_
#define XLS_COMMON_TO_STRING_HELPERS_H_

#include <deque>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"

// Helpers for generating the string representation of: C++ primitive integral
// types and selected containers containing primitive integral types or
// user-defined types with a defined ToString() method. Note that these helper
// functions are needed since an AbslFormatConvert does not exist for these
// types.
//
// The generation of the string are performed recursively. As a result, when
// generating the string representation of a container, the string
// representation of the elements within the container are also generated.
//
// To leverage the recursive mechanism when performing generating the string
// representation for user-defined types such as Classes and Structs, the
// function generating the string representation must be a member of the Class
// or Struct and must have the following signature:
//
//     std::string ToString() const;
//
// In the following example, the 'UserDefinedType' struct has the ToString
// method as a member defined.
//   struct UserDefinedType {
//     int64_t integer;
//     bool boolean;
//     std::string ToString() const {
//       return absl::StrFormat("{integer: %s, boolean: %s}",
//                              ::xls::ToString(integer),
//                              ::xls::ToString(boolean));
//     }
//   };
//
// Note that the member of the 'UserDefinedType' struct leverages the ToString
// methods for the C++ primitive integral types.
//
// A sample invocation of the ToString using a container containing an
// user-defined type is as follows:
//
//   ToString(absl::Span<const UserDefinedType>{UserDefinedType{42, true},
//                                              UserDefinedType{1, false}}
//

namespace xls {

// Returns the lower case string representation of the boolean value ("true" or
// "false").
std::string ToString(bool value);

// Returns a string representation of a C++ primitive integral type.
template <typename T, typename std::enable_if<std::is_integral_v<T> &&
                                                  !std::is_same_v<T, bool>,
                                              T>::type* = nullptr>
std::string ToString(T value) {
  return std::to_string(value);
}

// Helper to determine whether type T has a const_iterator member. A type T with
// a const_iterator is assumed to be a container.
template <typename T>
struct has_const_iterator {
 private:
  template <typename C>
  static int8_t test(typename C::const_iterator*);
  template <typename C>
  static int16_t test(...);

 public:
  static constexpr bool value = sizeof(test<T>(0)) == sizeof(int8_t);
};
template <typename T>
inline constexpr bool has_const_iterator_v = has_const_iterator<T>::value;

// Returns a string representation of a container containing another container
// or a C++ primitive integral type.
template <typename T, template <class...> class Container,
          typename std::enable_if<
              (std::is_integral_v<T> || has_const_iterator_v<T>)&&(
                  std::is_same_v<Container<T>, absl::Span<const T>> ||
                  std::is_same_v<Container<T>, std::vector<T>> ||
                  std::is_same_v<Container<T>, std::deque<T>>),
              T>::type* = nullptr>
std::string ToString(const Container<T>& values) {
  std::vector<std::string> entries;
  entries.reserve(values.size());
  for (const T& value : values) {
    entries.push_back(ToString(value));
  }
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("[ ", content, " ]");
}

// Returns a string representation of a container containing a user-defined
// type with a defined ToString() method.
template <typename T, template <class...> class Container,
          typename std::enable_if<
              (!std::is_integral_v<T> && !has_const_iterator_v<T>)&&(
                  std::is_same_v<Container<T>, absl::Span<const T>> ||
                  std::is_same_v<Container<T>, std::vector<T>> ||
                  std::is_same_v<Container<T>, std::deque<T>>),
              T>::type* = nullptr>
std::string ToString(const Container<T>& values) {
  std::vector<std::string> entries;
  entries.reserve(values.size());
  for (const T& value : values) {
    entries.push_back(value.ToString());
  }
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("[ ", content, " ]");
}

}  // namespace xls

#endif  // XLS_COMMON_TO_STRING_HELPERS_H_
