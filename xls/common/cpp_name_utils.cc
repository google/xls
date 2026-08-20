// Copyright 2026 The XLS Authors
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

#include "xls/common/cpp_name_utils.h"

#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"

namespace xls {

bool IsCppKeyword(std::string_view identifier) {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> keywords({
      "alignas",       "alignof",        "and",
      "and_eq",        "asm",            "atomic_cancel",
      "atomic_commit", "atomic_noexcept", "auto",
      "bitand",        "bitor",          "bool",
      "break",         "case",           "catch",
      "char",          "char8_t",        "char16_t",
      "char32_t",      "class",          "compl",
      "concept",       "const",          "consteval",
      "constexpr",     "constinit",      "const_cast",
      "continue",      "co_await",       "co_return",
      "co_yield",      "decltype",       "default",
      "delete",        "do",             "double",
      "dynamic_cast",  "else",           "enum",
      "explicit",      "export",         "extern",
      "false",         "float",          "for",
      "friend",        "goto",           "if",
      "inline",        "int",            "long",
      "mutable",       "namespace",      "new",
      "noexcept",      "not",            "not_eq",
      "nullptr",       "operator",       "or",
      "or_eq",         "private",        "protected",
      "public",        "reflexpr",       "register",
      "reinterpret_cast", "requires",    "return",
      "short",         "signed",         "sizeof",
      "static",        "static_assert",  "static_cast",
      "struct",        "switch",         "synchronized",
      "template",      "this",           "thread_local",
      "throw",         "true",           "try",
      "typedef",       "typeid",         "typename",
      "union",         "unsigned",       "using",
      "virtual",       "void",           "volatile",
      "wchar_t",       "while",          "xor",
      "xor_eq",
  });
  return keywords->contains(identifier);
}

}  // namespace xls
