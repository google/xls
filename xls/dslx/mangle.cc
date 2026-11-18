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

#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {
namespace {

// Converts the given InterpValue into a string form appropriate for name
// mangling: we need a means of disambiguating value- or struct-parameterized
// function names, and so we need to be able to represent instantiation values.
absl::StatusOr<std::string> MangleInterpValue(const InterpValue& value) {
  if (value.IsBits() || value.IsEnum()) {
    auto bits_value = InterpValue::MakeBits(/*is_signed=*/value.IsSigned(),
                                            value.GetBitsOrDie());
    std::string s = bits_value.ToString(/*humanize=*/true,
                                        FormatPreference::kUnsignedDecimal);
    // Negatives are not valid characters in IR symbols so replace leading '-'
    // with 'm'.
    return absl::StrReplaceAll(s, {{"-", "m"}});
  }

  if (value.IsTypeReference()) {
    return value.GetTypeReferenceSpecializationName();
  }

  CHECK(value.IsArray() || value.IsTuple())
      << "Only bits, enums, arrays, or tuples can be name-mangled.";
  std::vector<std::string> members;
  for (const auto& member : value.GetValuesOrDie()) {
    XLS_ASSIGN_OR_RETURN(std::string mangled_member, MangleInterpValue(member));
    members.push_back(std::move(mangled_member));
  }

  // Preserve the established spelling of aggregate value arguments.
  return absl::StrFormat("__%d%s__", members.size(),
                         absl::StrJoin(members, "_"));
}

std::string Frame(std::string_view value) {
  return absl::StrCat(value.size(), "_", value);
}

}  // namespace

// LINT.IfChange
absl::StatusOr<std::string> MangleDslxName(
    std::string_view module_name, std::string_view function_name,
    CallingConvention convention, const absl::btree_set<std::string>& free_keys,
    const ParametricEnv& parametric_env, std::string_view scope) {
  absl::btree_set<std::string> parametric_env_keys;
  std::vector<std::string> parametric_env_values;
  bool has_type_parametric = false;
  for (const ParametricEnvItem& item : parametric_env.bindings()) {
    parametric_env_keys.insert(item.identifier);
    const InterpValue& value = item.value;
    XLS_ASSIGN_OR_RETURN(std::string mangled_value, MangleInterpValue(value));
    has_type_parametric |= value.IsTypeReference();
    parametric_env_values.push_back(std::move(mangled_value));
  }
  absl::btree_set<std::string> difference;
  absl::c_set_difference(free_keys, parametric_env_keys,
                         std::inserter(difference, difference.begin()));
  if (!difference.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not enough parametric bindings to convert function "
                        "'%s'; need {%s} got {%s}",
                        function_name, absl::StrJoin(free_keys, ", "),
                        absl::StrJoin(parametric_env_keys, ", ")));
  }

  std::string convention_str;
  if (convention == CallingConvention::kImplicitToken) {
    // Note that this is an implicit token threaded call target.
    convention_str = "itok__";
  }
  std::string module_name_str = absl::StrReplaceAll(module_name, {{".", "_"}});

  std::string scope_str = "";
  if (!scope.empty()) {
    scope_str = absl::StrCat(absl::StrReplaceAll(scope,
                                                 {
                                                     {">", "_"},
                                                     {"<", "_"},
                                                     {"/", "_"},
                                                     {":", "_"},
                                                     {"-", "_"},
                                                     {".", "_"},
                                                     {",", "__"},
                                                     {" ", ""},
                                                 }),
                             "__");
  }

  std::string suffix;
  if (has_type_parametric) {
    // Framing the complete name makes argument and function-name boundaries
    // unambiguous even when a type's encoding contains underscores. The
    // single-leading-underscore namespace cannot collide with the existing
    // double-leading-underscore namespace for ordinary and value-only names.
    std::vector<std::string> framed_values;
    for (size_t i = 0; i < parametric_env_values.size(); ++i) {
      const InterpValue& value = parametric_env.bindings()[i].value;
      framed_values.push_back(absl::StrCat(value.IsTypeReference() ? "T" : "V",
                                           Frame(parametric_env_values[i])));
    }
    std::string function_name_str(function_name);
    if (convention == CallingConvention::kProcNext) {
      convention_str = "pnext__";
      function_name_str =
          absl::StrReplaceAll(function_name_str, {{":", "_"}, {"->", "__"}});
    }
    return absl::StrCat(
        "_0T__", convention_str, Frame(module_name_str), "_", Frame(scope_str),
        "_", Frame(function_name_str), "__", absl::StrJoin(framed_values, "_"),
        convention == CallingConvention::kProcNext ? "_next" : "");
  }
  if (!parametric_env_values.empty()) {
    suffix = absl::StrCat("__", absl::StrJoin(parametric_env_values, "_"));
  }
  std::string mangled_name =
      absl::StrFormat("__%s%s__%s%s%s", convention_str, module_name_str,
                      scope_str, function_name, suffix);
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
