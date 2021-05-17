// Copyright 2021 The XLS Authors
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

#include "xls/dslx/mangle.h"

#include "absl/strings/str_replace.h"
#include "xls/common/status/ret_check.h"

namespace xls::dslx {

absl::StatusOr<std::string> MangleDslxName(
    absl::string_view module_name, absl::string_view function_name,
    CallingConvention convention, const absl::btree_set<std::string>& free_keys,
    const SymbolicBindings* symbolic_bindings) {
  absl::btree_set<std::string> symbolic_bindings_keys;
  // TODO(rspringer): 2021-03-25 This can't yet be InterpValue'd, since we'd
  // have to be able to mangle array/tuple types, which is NYI.
  std::vector<int64_t> symbolic_bindings_values;
  if (symbolic_bindings != nullptr) {
    for (const SymbolicBinding& item : symbolic_bindings->bindings()) {
      symbolic_bindings_keys.insert(item.identifier);
      int64_t value = item.value.IsSigned()
                          ? item.value.GetBitValueInt64().value()
                          : item.value.GetBitValueUint64().value();
      symbolic_bindings_values.push_back(value);
    }
  }
  absl::btree_set<std::string> difference;
  absl::c_set_difference(free_keys, symbolic_bindings_keys,
                         std::inserter(difference, difference.begin()));
  if (!difference.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not enough symbolic bindings to convert function "
                        "'%s'; need {%s} got {%s}",
                        function_name, absl::StrJoin(free_keys, ", "),
                        absl::StrJoin(symbolic_bindings_keys, ", ")));
  }

  std::string convention_str;
  if (convention == CallingConvention::kImplicitToken) {
    // Note that this is an implicit token threaded call target.
    convention_str = "itok__";
  }
  std::string module_name_str = absl::StrReplaceAll(module_name, {{".", "_"}});
  if (symbolic_bindings_values.empty()) {
    return absl::StrFormat("__%s%s__%s", convention_str, module_name_str,
                           function_name);
  }
  std::string suffix = absl::StrJoin(symbolic_bindings_values, "_");
  return absl::StrFormat("__%s%s__%s__%s", convention_str, module_name_str,
                         function_name, suffix);
}

}  // namespace xls::dslx
