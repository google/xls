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
namespace {

// Converts the given InterpValue into a string form appropriate for name
// mangling: we need a means of disambiguating value- or struct-parameterized
// function names, and so we need to be able to represent instantiation values.
std::string MangleInterpValue(const InterpValue& value) {
  if (value.IsBits() || value.IsEnum()) {
    auto bits_value = InterpValue::MakeBits(/*is_signed=*/value.IsSigned(),
                                            value.GetBitsOrDie());
    std::string s = bits_value.ToString(/*humanize=*/true,
                                        FormatPreference::kUnsignedDecimal);
    // Negatives are not valid characters in IR symbols so replace leading '-'
    // with 'm'.
    return absl::StrReplaceAll(s, {{"-", "m"}});
  }

  XLS_CHECK(value.IsArray() || value.IsTuple())
      << "Only bits, enums, arrays, or tuples can be name-mangled.";
  std::vector<std::string> members;
  for (const auto& member : value.GetValuesOrDie()) {
    members.push_back(MangleInterpValue(member));
  }

  // Since functions can't be overloaded, there's no risk of name collision:
  // each parametric in a Function specification has a dedicated location in the
  // mangled name. Functions with identical parametric values are identical.
  return absl::StrFormat("__%d%s__", members.size(),
                         absl::StrJoin(members, "_"));
}

}  // namespace

// LINT.IfChange
absl::StatusOr<std::string> MangleDslxName(
    std::string_view module_name, std::string_view function_name,
    CallingConvention convention, const absl::btree_set<std::string>& free_keys,
    const SymbolicBindings* symbolic_bindings) {
  absl::btree_set<std::string> symbolic_bindings_keys;
  std::vector<std::string> symbolic_bindings_values;
  if (symbolic_bindings != nullptr) {
    for (const SymbolicBinding& item : symbolic_bindings->bindings()) {
      symbolic_bindings_keys.insert(item.identifier);
      const InterpValue& value = item.value;
      symbolic_bindings_values.push_back(MangleInterpValue(value));
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

  std::string suffix;
  if (!symbolic_bindings_values.empty()) {
    suffix = absl::StrCat("__", absl::StrJoin(symbolic_bindings_values, "_"));
  }
  std::string mangled_name = absl::StrFormat(
      "__%s%s__%s%s", convention_str, module_name_str, function_name, suffix);
  if (convention == CallingConvention::kProcNext) {
    // Note that the identifier for the next function of a proc is used to
    // identify the proc in the conversion. The config function of a proc is
    // absorbed in the conversion, as a result it is never referenced.
    // Thus, it does not need to be mangled.
    mangled_name = absl::StrCat(
        absl::StrReplaceAll(mangled_name, {{":", "_"}, {"->", "__"}}), "_next");
  }
  return mangled_name;
}
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

}  // namespace xls::dslx
