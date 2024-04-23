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

#ifndef XLS_COMMON_COMPARISON_UTILS_H_
#define XLS_COMMON_COMPARISON_UTILS_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/to_string_utils.h"
#include "xls/common/type_traits_utils.h"

// Utilities for comparing C++ primitive types, std::pair and containers with a
// const_iterator and size() members. The comparisons return a human readable
// string useful during debugging.

namespace xls {

// Utilities for comparing C++ primitive types, std::pair and containers with a
// const_iterator and size() members.
//
// The comparisons are performed recursively. As a result, when comparing a
// container, the elements within the container are also compared.
//
// To leverage the recursive mechanism when performing comparisons for
// user-defined types such as Classes and Structs, the comparison function must
// have the following signature, where 'MyClassOrStruct' is a user-defined class
// or struct:
//
//     std::string Compare(std::string_view element_name,
//                         const MyClassOrStruct& expected,
//                         const MyClassOrStruct& computed);
//
// Moreover, the function must be placed in the 'xls' namespace.
//
// In addition, when defining a user-defined class or struct, the '!=' operator
// for the user-defined class or struct must be defined in the 'xls' namespace.
// The following is an example of a user-defined struct declaration:
//     namespace userspace {
//       struct UserStruct {
//        int64_t integer_value;
//        bool boolean_value;
//       };
//     }  // namespace userspace
//     namespace xls {
//      bool operator!=(const ::userspace::UserStruct& lhs,
//                       const ::userspace::UserStruct& rhs);
//     }  // namespace xls
//
// In practice, the user may define the '==' operator and define a '!=' operator
// that negates the result from the '==' operator. For example, for the
// UserStruct struct may have the following definition:
//     namespace xls {
//       bool operator==(const ::userspace::UserStruct& lhs,
//                       const ::userspace::UserStruct& rhs) {
//         return lhs.integer_value == rhs.integer_value &&
//                lhs.boolean_value == rhs.boolean_value;
//       }
//       bool operator!=(const ::userspace::UserStruct& lhs,
//                       const ::userspace::UserStruct& rhs) {
//         return !(lhs == rhs);
//       }
//     }  // namespace xls
//
// Given the UserStruct struct defined above, a sample comparison function would
// be as follows:
//
//     namespace xls {
//       std::string Compare(std::string_view element_name,
//                           const ::userspace::UserStruct& expected,
//                           const ::userspace::UserStruct& computed) {
//         std::string comparison;
//         std::string ref_element_name;
//         if (!element_name.empty()) {
//           ref_element_name = absl::StrCat(element_name, ".");
//         }
//         absl::StrAppend(
//           &comparison,
//           absl::StrFormat("%s",
//             ::xls::Compare(absl::StrCat(ref_element_name, "integer_value"),
//                            expected.integer_value, computed.integer_value)));
//         absl::StrAppend(
//           &comparison,
//           absl::StrFormat("%s",
//             ::xls::Compare(absl::StrCat(ref_element_name, "boolean_value"),
//                            expected.boolean_value, computed.boolean_value)));
//         return comparison;
//       }
//     }  // namespace xls
//
// Note that the comparison function leverages the comparison function for C++
// primitives.

// Compares integral types.
template <typename T,
          typename std::enable_if<std::is_integral_v<T>, T>::type* = nullptr>
std::string Compare(std::string_view element_name, T expected, T computed) {
  if (computed == expected) {
    return "";
  }
  return absl::StrFormat("Element %s differ: expected (%s), got (%s).\n",
                         element_name, ToString(expected), ToString(computed));
}

// Compares std::pair types.
template <typename T1, typename T2>
std::string Compare(std::string_view element_name,
                    const std::pair<T1, T2>& expected,
                    const std::pair<T1, T2>& computed) {
  std::string comparison;
  std::string ref_element_name = "";
  if (!element_name.empty()) {
    ref_element_name = absl::StrCat(element_name, ".");
  }
  absl::StrAppend(
      &comparison,
      absl::StrFormat("%s",
                      ::xls::Compare(absl::StrCat(ref_element_name, "first"),
                                     expected.first, computed.first)));
  absl::StrAppend(
      &comparison,
      absl::StrFormat("%s",
                      ::xls::Compare(absl::StrCat(ref_element_name, "second"),
                                     expected.second, computed.second)));
  return comparison;
}

// Compares a container.
template <typename T, template <class...> class Container,
          typename std::enable_if<has_const_iterator_v<Container<T>> &&
                                      has_member_size_v<Container<T>>,
                                  T>::type* = nullptr>
std::string Compare(std::string_view element_name,
                    const Container<T>& expected,
                    const Container<T>& computed) {
  if (computed.size() != expected.size()) {
    return absl::StrFormat(
        "Size of element %s differ: expected (%s), got (%s).\n", element_name,
        ToString(expected.size()), ToString(computed.size()));
  }
  std::string comparison;
  int64_t i = 0;
  for (typename Container<T>::const_iterator computed_it = computed.cbegin(),
                                             expected_it = expected.cbegin();
       computed_it != computed.cend(); ++computed_it, ++expected_it) {
    if (*computed_it != *expected_it) {
      absl::StrAppend(&comparison,
                      Compare(absl::StrCat(element_name, "[", i, "]"),
                              *expected_it, *computed_it));
    }
    i++;
  }
  return comparison;
}

}  // namespace xls

#endif  // XLS_COMMON_COMPARISON_UTILS_H_
