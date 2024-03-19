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

#ifndef XLS_COMMON_VISITOR_H_
#define XLS_COMMON_VISITOR_H_

namespace xls {

// A utility for concisely defining std::variant visitors. Example usage:
//
// std::variant<int, string> v = 3;
// absl::visit(Visitor{
//   [](int) { LOG(INFO) << "It's an int"; }
//   [](string&) { LOG(INFO) << "It's a string"; }
// }, v);
//
// See https://en.cppreference.com/w/cpp/utility/variant/visit example #4.
template <class... Ts>
struct Visitor : Ts... {
  using Ts::operator()...;
};

// This is a user-defined deduction guide. It allows constructing visitors using
// Visitor{[](X x) {}, [](Y y) {}} syntax. This works through aggregate
// initialization, since all Visitor types are aggregates.
template <class... Ts>
Visitor(Ts...) -> Visitor<Ts...>;

}  // namespace xls

#endif  // XLS_COMMON_VISITOR_H_
