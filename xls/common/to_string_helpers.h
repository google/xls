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
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/common/type_traits_helpers.h"

// Helpers for generating the string representation of: C++ primitive integral
// types and containers with a const_iterator and size() members that contain
// primitive integral types or user-defined types with a defined ToString()
// method. Note that these helper functions are needed since an
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

// TODO(vmirian): 7-25-2022 Enable ToString for std::pair to support std::map
// and std::multimap, std::unordered_map, std::unordered_multimap.
// TODO(vmirian): 7-25-2022 Add support for std::array.

// Returns a string representation of a container containing a type without a
// defined ToString() method.
template <typename T, template <class...> class Container,
          typename std::enable_if<
              (std::is_integral_v<T> ||
               has_const_iterator_v<T>)&&has_const_iterator_v<Container<T>> &&
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
          typename std::enable_if<!std::is_integral_v<T> &&
                                      !has_const_iterator_v<T> &&
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
