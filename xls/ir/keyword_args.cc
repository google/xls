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

#include "xls/ir/keyword_args.h"

#include <concepts>  // NOLINT(misc-include-cleaner)  clang-tidy is confused.
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"

namespace xls {

namespace {
template <typename T, typename U>
  requires std::invocable<U, T const&> &&
           std::convertible_to<std::invoke_result_t<U, T const&>,
                               std::string_view>
absl::StatusOr<std::vector<Value>> KeywordArgsToPositionalImpl(
    absl::Span<T> named_arg_span, const U& get_name,
    std::string_view function_name,
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  VLOG(2) << "Interpreting function " << function_name << " with arguments:";

  absl::flat_hash_map<std::string_view, int64_t> param_indices;

  int64_t param_index = 0;
  for (const T& named_arg : named_arg_span) {
    std::string_view name = get_name(named_arg);
    if (!kwargs.contains(name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing argument '%s' to invocation of function '%s'", name,
          function_name));
    }
    auto [_, inserted] = param_indices.insert({name, param_index++});
    XLS_RET_CHECK(inserted) << "Duplicate argument name " << name;
  }

  XLS_RET_CHECK_EQ(named_arg_span.size(), kwargs.size())
      << "Too many arguments in invocation of function " << function_name;

  std::vector<Value> positional_args;
  positional_args.resize(kwargs.size());
  for (const auto& [name, value] : kwargs) {
    VLOG(2) << "  " << name << " = " << value;
    int64_t param_index = param_indices.at(name);
    positional_args[param_index] = value;
  }

  return positional_args;
}
}  // namespace

absl::StatusOr<std::vector<Value>> KeywordArgsToPositional(
    const FunctionBase& function,
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  return KeywordArgsToPositionalImpl(
      function.params(),
      [](Param* param) -> std::string_view { return param->name(); },
      function.name(), kwargs);
}

absl::StatusOr<std::vector<Value>> KeywordArgsToPositional(
    absl::Span<std::string const> param_names,
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  return KeywordArgsToPositionalImpl(
      param_names,
      [](std::string_view name) -> std::string_view { return name; },
      "<unknown>", kwargs);
}

}  // namespace xls
