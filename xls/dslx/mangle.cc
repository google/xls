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

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_specialization.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {
namespace {

using type_system_internal::BindingKind;
using type_system_internal::NominalPart;
using type_system_internal::SpecializationKind;
using type_system_internal::SpecializationPart;

// MangleType grammar. N is a nonnegative decimal integer terminated by '_'.
// I is N followed by exactly that many escaped identifier bytes. [X] is a
// count N followed by that many Xs. T is one of:
//
//   u N | s N                    unsigned or signed bits
//   a N T | q [T]                array or tuple
//   e I I (u | s) N              enum (module, declaration, underlying bits)
//   (r | p) I I [I B]            struct or proc (bindings sorted by name)
//   z I I [(u | t | r) [T]]      sum (unit, tuple, or struct variants)
//   f [T] T                      function (parameters, return)
//   (ci | co) T | m T            channel or meta-type
//   d I | token                  module or token
//
// A nominal binding B is t T (type), v N H (bits or enum, width then
// unsigned hexadecimal digits terminated by '_'), q [B] (tuple), or a [B]
// (array). Each child ends where its grammar says, so it needs no length
// prefix. For example, u32[4] is a4_u32_.
std::string_view KindTag(SpecializationKind kind) {
  switch (kind) {
    case SpecializationKind::kUnsignedBits:
    case SpecializationKind::kUnitVariant:
      return "u";
    case SpecializationKind::kSignedBits:
      return "s";
    case SpecializationKind::kEnum:
      return "e";
    case SpecializationKind::kProc:
      return "p";
    case SpecializationKind::kStruct:
    case SpecializationKind::kStructVariant:
      return "r";
    case SpecializationKind::kSum:
      return "z";
    case SpecializationKind::kTupleVariant:
      return "t";
    case SpecializationKind::kTuple:
      return "q";
    case SpecializationKind::kArray:
      return "a";
    case SpecializationKind::kFunction:
      return "f";
    case SpecializationKind::kChannelIn:
      return "ci";
    case SpecializationKind::kChannelOut:
      return "co";
    case SpecializationKind::kMeta:
      return "m";
    case SpecializationKind::kModule:
      return "d";
    case SpecializationKind::kToken:
      return "token";
  }
}

// LINT.IfChange
std::string Frame(std::string_view value) {
  return absl::StrCat(value.size(), "_", value);
}

// Each non-alphanumeric byte becomes _HH (uppercase hex), including '_'
// so a literal "_2E" remains distinct from ".".
std::string FrameIdentifier(std::string_view identifier) {
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string escaped;
  for (unsigned char c : identifier) {
    if (absl::ascii_isalnum(c)) {
      escaped.push_back(c);
    } else {
      escaped.push_back('_');
      escaped.push_back(kHex[c >> 4]);
      escaped.push_back(kHex[c & 0xf]);
    }
  }
  return Frame(escaped);
}
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

absl::Status AppendBinding(std::string& result, const InterpValue& value) {
  switch (type_system_internal::GetBindingKind(value)) {
    case BindingKind::kType: {
      XLS_ASSIGN_OR_RETURN(std::string name,
                           value.GetTypeReferenceSpecializationName());
      absl::StrAppend(&result, "t", name);
      return absl::OkStatus();
    }
    case BindingKind::kBits: {
      const auto& bits = value.GetBitsOrDie();
      absl::StrAppend(&result, "v", bits.bit_count(), "_",
                      BitsToRawDigits(bits, FormatPreference::kPlainHex), "_");
      return absl::OkStatus();
    }
    case BindingKind::kTuple:
    case BindingKind::kArray: {
      XLS_ASSIGN_OR_RETURN(int64_t length, value.GetLength());
      absl::StrAppend(&result, value.IsTuple() ? "q" : "a", length, "_");
      for (int64_t i = 0; i < length; ++i) {
        XLS_ASSIGN_OR_RETURN(InterpValue element, value.Index(i));
        XLS_RETURN_IF_ERROR(AppendBinding(result, element));
      }
      return absl::OkStatus();
    }
    case BindingKind::kUnsupported:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported nominal specialization binding: ",
                       TagToString(value.tag())));
  }
}

absl::Status AppendType(std::string& result, const Type& type) {
  XLS_ASSIGN_OR_RETURN(std::vector<SpecializationPart> parts,
                       type_system_internal::DescribeSpecialization(type));
  for (const SpecializationPart& part : parts) {
    if (const auto* kind = std::get_if<SpecializationKind>(&part)) {
      result.append(KindTag(*kind));
    } else if (const auto* number = std::get_if<int64_t>(&part)) {
      if (*number < 0) {
        return absl::InvalidArgumentError("Negative specialization dimension");
      }
      absl::StrAppend(&result, *number, "_");
    } else if (const auto* identifier = std::get_if<std::string_view>(&part)) {
      result.append(FrameIdentifier(*identifier));
    } else if (const auto* nominal = std::get_if<NominalPart>(&part)) {
      result.append(FrameIdentifier(nominal->module->name()));
      if (nominal->identifier.has_value()) {
        result.append(FrameIdentifier(*nominal->identifier));
      }
    } else if (const auto* child = std::get_if<const Type*>(&part)) {
      XLS_RETURN_IF_ERROR(AppendType(result, **child));
    } else {
      XLS_RETURN_IF_ERROR(
          AppendBinding(result, *std::get<const InterpValue*>(part)));
    }
  }
  return absl::OkStatus();
}

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

}  // namespace

absl::StatusOr<std::string> MangleType(const Type& type) {
  std::string result;
  XLS_RETURN_IF_ERROR(AppendType(result, type));
  return result;
}

// LINT.IfChange
absl::StatusOr<std::string> MangleDslxName(
    std::string_view module_name, std::string_view function_name,
    CallingConvention convention, const absl::btree_set<std::string>& free_keys,
    const ParametricEnv& parametric_env, std::string_view scope) {
  absl::btree_set<std::string> parametric_env_keys;
  std::vector<std::string> parametric_env_values;
  bool has_type_parametric = absl::c_any_of(
      parametric_env.bindings(), [](const ParametricEnvItem& item) {
        return item.value.IsTypeReference();
      });
  for (const ParametricEnvItem& item : parametric_env.bindings()) {
    parametric_env_keys.insert(item.identifier);
    const InterpValue& value = item.value;
    XLS_ASSIGN_OR_RETURN(std::string mangled_value, MangleInterpValue(value));
    if (has_type_parametric) {
      mangled_value = absl::StrCat(value.IsTypeReference() ? "T" : "V",
                                   Frame(mangled_value));
    }
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
  std::string suffix;
  if (!parametric_env_values.empty()) {
    suffix = absl::StrCat("__", absl::StrJoin(parametric_env_values, "_"));
  }

  if (has_type_parametric) {
    // _0T__ separates specializations with type arguments from the ordinary
    // __ namespace. Each component has a byte length, including an empty
    // scope. T and V introduce a mangled type and a mangled value respectively.
    if (convention == CallingConvention::kProcNext) {
      convention_str = "pnext__";
      suffix += "_next";
    }
    return absl::StrCat("_0T__", convention_str, FrameIdentifier(module_name),
                        "_", FrameIdentifier(scope), "_",
                        FrameIdentifier(function_name), suffix);
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
