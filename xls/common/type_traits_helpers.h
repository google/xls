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

#ifndef XLS_COMMON_TYPE_TRAITS_HELPERS_H_
#define XLS_COMMON_TYPE_TRAITS_HELPERS_H_

#include <cstdint>
#include <type_traits>
#include <variant>

namespace xls {

typedef int8_t YesType;
typedef int16_t NoType;

// Helper to determine whether a const_iterator member exists.
template <typename T>
struct has_const_iterator {
 private:
  template <typename C>
  static YesType Test(typename C::const_iterator*);
  template <typename C>
  static NoType Test(...);

 public:
  static constexpr bool value = sizeof(Test<T>(0)) == sizeof(YesType);
};
template <typename T>
inline constexpr bool has_const_iterator_v = has_const_iterator<T>::value;

// Helper to determine whether a size member exists.
template <typename T>
struct has_member_size {
 private:
  template <typename C>
  static YesType Test(decltype(&C::size));
  template <typename C>
  static NoType Test(...);

 public:
  static constexpr bool value = sizeof(Test<T>(0)) == sizeof(YesType);
};
template <typename T>
inline constexpr bool has_member_size_v = has_member_size<T>::value;

// Helper to determine whether a ToString member exists.
template <typename T>
struct has_member_to_string {
 private:
  template <typename C>
  static YesType Test(decltype(&C::ToString));
  template <typename C>
  static NoType Test(...);

 public:
  static constexpr bool value = sizeof(Test<T>(0)) == sizeof(YesType);
};
template <typename T>
inline constexpr bool has_member_to_string_v = has_member_to_string<T>::value;

// Helper to determine if a Subject Type is a type defined within a
// std::variant.
template <typename...>
struct is_one_of : std::false_type {};
template <typename Subject>
struct is_one_of<Subject> : std::false_type {};
template <typename Subject, typename Type>
struct is_one_of<Subject, std::variant<Type>> : std::is_same<Subject, Type> {};
template <typename Subject, typename Type, typename... Types>
struct is_one_of<Subject, std::variant<Type, Types...>>
    : std::conditional_t<std::is_same_v<Subject, Type>, std::true_type,
                         is_one_of<Subject, std::variant<Types...>>> {};

}  // namespace xls

#endif  // XLS_COMMON_TYPE_TRAITS_HELPERS_H_
