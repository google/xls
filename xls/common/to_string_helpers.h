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

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/common/type_traits_helpers.h"

// Helpers for generating the string representation of: C++ primitive integral
// types, std::pair and containers with a const_iterator and size() members that
// contain primitive integral types or user-defined types with a defined
// ToString() method. Note that these helper functions are needed since an
// AbslFormatConvert does not exist for these types.
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

// Returns a string representation of a std::pair containing elements without a
// ToString method.
template <typename T1, typename T2,
          typename std::enable_if<!has_member_to_string_v<T1> &&
                                      !has_member_to_string_v<T2>,
                                  T1>::type* = nullptr>
std::string ToString(const std::pair<T1, T2>& value) {
  std::vector<std::string> entries = {ToString(value.first),
                                      ToString(value.second)};
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("{ ", content, " }");
}

// Returns a string representation of a std::pair with the first element
// not containing a ToString method and the second element containing a ToString
// method.
template <typename T1, typename T2,
          typename std::enable_if<!has_member_to_string_v<T1> &&
                                      has_member_to_string_v<T2>,
                                  T1>::type* = nullptr>
std::string ToString(const std::pair<T1, T2>& value) {
  std::vector<std::string> entries = {ToString(value.first),
                                      value.second.ToString()};
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("{ ", content, " }");
}

// Returns a string representation of a std::pair with the first element
// containing a ToString method and the second element not containing a ToString
// method.
template <typename T1, typename T2,
          typename std::enable_if<has_member_to_string_v<T1> &&
                                      !has_member_to_string_v<T2>,
                                  T1>::type* = nullptr>
std::string ToString(const std::pair<T1, T2>& value) {
  std::vector<std::string> entries = {value.first.ToString(),
                                      ToString(value.second)};
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("{ ", content, " }");
}

// Returns a string representation of a std::pair containing elements with a
// ToString method.
template <typename T1, typename T2,
          typename std::enable_if<has_member_to_string_v<T1> &&
                                      has_member_to_string_v<T2>,
                                  T1>::type* = nullptr>
std::string ToString(const std::pair<T1, T2>& value) {
  std::vector<std::string> entries = {value.first.ToString(),
                                      value.second.ToString()};
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("{ ", content, " }");
}

// TODO(vmirian): 7-25-2022 Enable support for std::map, std::multimap,
// std::unordered_map and std::unordered_multimap.
// TODO(vmirian): 7-25-2022 Add support for std::array.

// Returns a string representation of a container containing a type without a
// defined ToString() method.
template <typename T, template <class...> class Container,
          typename std::enable_if<!has_member_to_string_v<T> &&
                                      has_const_iterator_v<Container<T>> &&
                                      has_member_size_v<Container<T>>,
                                  T>::type* = nullptr>
std::string ToString(const Container<T>& values) {
  std::vector<std::string> entries;
  entries.reserve(values.size());
  for (typename Container<T>::const_iterator it = values.cbegin();
       it != values.cend(); ++it) {
    entries.push_back(ToString(*it));
  }
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("[ ", content, " ]");
}

// Returns a string representation of a container containing a type with a
// defined ToString() method.
template <typename T, template <class...> class Container,
          typename std::enable_if<has_member_to_string_v<T> &&
                                      has_const_iterator_v<Container<T>> &&
                                      has_member_size_v<Container<T>>,
                                  T>::type* = nullptr>
std::string ToString(const Container<T>& values) {
  std::vector<std::string> entries;
  entries.reserve(values.size());
  for (typename Container<T>::const_iterator it = values.cbegin();
       it != values.cend(); ++it) {
    entries.push_back(it->ToString());
  }
  std::string content = absl::StrJoin(entries, ", ");
  return absl::StrCat("[ ", content, " ]");
}

}  // namespace xls

#endif  // XLS_COMMON_TO_STRING_HELPERS_H_
