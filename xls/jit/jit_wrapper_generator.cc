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
#include "xls/jit/jit_wrapper_generator.h"

#include <algorithm>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"

namespace xls {
namespace {

// Matches the parameter signature for an "implicit token/activation taking"
// function.
bool MatchesImplicitToken(absl::Span<Param* const> params) {
  if (params.size() < 2) {
    return false;
  }
  Type* p0_type = params[0]->GetType();
  Type* p1_type = params[1]->GetType();
  return p0_type->IsToken() && p1_type->IsBits() &&
         p1_type->GetFlatBitCount() == 1;
}

// Determines the "effective signature" for XLS IR function "f".
//
// Returns [params, return_type], and optionally informs (via outparam pointer)
// whether the "implicit token" calling convention was stripped off of the
// signature to produce the return value.
//
// Since the JIT wrapper generally wants to expose fallible functions with their
// *meaningful* signature, stripping off the "implicit token" portions of the
// signature is generally helpful for callers.
std::pair<absl::Span<Param* const>, Type*> GetSignature(
    const Function& f, bool* implicit_token_convention = nullptr) {
  absl::Span<Param* const> params = f.params();
  Type* return_type = f.return_value()->GetType();
  bool has_implicit_token = MatchesImplicitToken(params);
  if (implicit_token_convention != nullptr) {
    *implicit_token_convention = has_implicit_token;
  }
  if (has_implicit_token) {
    CHECK(return_type->IsTuple() && return_type->AsTupleOrDie()->size() == 2)
        << "'Implicit token' calling convention requires return type to be a "
           "tuple of the form `(token, real_retval)`; got: "
        << return_type->ToString();
    params = params.subspan(2);
    return_type = return_type->AsTupleOrDie()->element_type(1);
  }
  return {params, return_type};
}

// Returns true if the given type can be represented as a native unsigned
// integer type (uint8_t, uint16_t, ...) and sets "enclosing_type" to the name
// of that type (as a string).
// Not a "match", per se, since we're finding the next enclosing native type
// (instead of strict matching), but for consistency with other match
// operations, we'll keep the MatchUint name.
bool MatchUint(const Type& type, std::string* enclosing_type) {
  if (!type.IsBits()) {
    return false;
  }

  int bit_count = type.GetFlatBitCount();
  if (bit_count <= 8) {
    *enclosing_type = "uint8_t";
    return true;
  }
  if (bit_count <= 16) {
    *enclosing_type = "uint16_t";
    return true;
  }
  if (bit_count <= 32) {
    *enclosing_type = "uint32_t";
    return true;
  }
  if (bit_count <= 64) {
    *enclosing_type = "uint64_t";
    return true;
  }

  return false;
}

// Returns true if the given type matches the C float type layout.
bool MatchFloat(const Type& type) {
  if (!type.IsTuple()) {
    return false;
  }

  const TupleType* tuple_type = type.AsTupleOrDie();
  auto element_types = tuple_type->element_types();
  if (element_types.size() != 3) {
    return false;
  }

  if (element_types[0]->IsBits() && element_types[0]->GetFlatBitCount() == 1 &&
      element_types[1]->IsBits() && element_types[1]->GetFlatBitCount() == 8 &&
      element_types[2]->IsBits() && element_types[2]->GetFlatBitCount() == 23) {
    return true;
  }

  return false;
}

// Returns true if the given type matches the C float type layout.
bool MatchDouble(const Type& type) {
  if (!type.IsTuple()) {
    return false;
  }

  const TupleType* tuple_type = type.AsTupleOrDie();
  auto element_types = tuple_type->element_types();
  if (element_types.size() != 3) {
    return false;
  }

  if (element_types[0]->IsBits() && element_types[0]->GetFlatBitCount() == 1 &&
      element_types[1]->IsBits() && element_types[1]->GetFlatBitCount() == 11 &&
      element_types[2]->IsBits() && element_types[2]->GetFlatBitCount() == 52) {
    return true;
  }

  return false;
}

// Returns the string representation of the packed view type corresponding to
// the given Type.
std::string PackedTypeString(const Type& type) {
  if (type.IsBits()) {
    return absl::StrCat("xls::PackedBitsView<", type.GetFlatBitCount(), ">");
  }
  if (type.IsArray()) {
    const ArrayType* array_type = type.AsArrayOrDie();
    std::string element_type_str =
        PackedTypeString(*array_type->element_type());
    return absl::StrFormat("xls::PackedArrayView<%s, %d>", element_type_str,
                           array_type->size());
  }  // Is tuple!
  CHECK(type.IsTuple()) << type.ToString();
  const TupleType* tuple_type = type.AsTupleOrDie();
  std::vector<std::string> element_type_strs;
  for (const Type* element_type : tuple_type->element_types()) {
    element_type_strs.push_back(PackedTypeString(*element_type));
  }
  return absl::StrFormat("xls::PackedTupleView<%s>",
                         absl::StrJoin(element_type_strs, ", "));
}

// Returns the string representation of the unpacked view type corresponding to
// the given Type.
std::string UnpackedTypeString(const Type& type) {
  if (type.IsBits()) {
    return absl::StrCat("xls::BitsView<", type.GetFlatBitCount(), ">");
  }
  if (type.IsArray()) {
    const ArrayType* array_type = type.AsArrayOrDie();
    std::string element_type_str =
        UnpackedTypeString(*array_type->element_type());
    return absl::StrFormat("xls::ArrayView<%s, %d>", element_type_str,
                           array_type->size());
  }  // Is tuple!
  CHECK(type.IsTuple()) << type.ToString();
  const TupleType* tuple_type = type.AsTupleOrDie();
  std::vector<std::string> element_type_strs;
  for (const Type* element_type : tuple_type->element_types()) {
    element_type_strs.push_back(UnpackedTypeString(*element_type));
  }
  return absl::StrFormat("xls::TupleView<%s>",
                         absl::StrJoin(element_type_strs, ", "));
}

// Returns the string representation of the unpacked mutable view type
// corresponding to the given Type.
std::string UnpackedMutableTypeString(const Type& type) {
  if (type.IsBits()) {
    return absl::StrCat("xls::MutableBitsView<", type.GetFlatBitCount(), ">");
  }
  if (type.IsArray()) {
    const ArrayType* array_type = type.AsArrayOrDie();
    std::string element_type_str =
        UnpackedMutableTypeString(*array_type->element_type());
    return absl::StrFormat("xls::MutableArrayView<%s, %d>", element_type_str,
                           array_type->size());
  }  // Is tuple!
  CHECK(type.IsTuple()) << type.ToString();
  const TupleType* tuple_type = type.AsTupleOrDie();
  std::vector<std::string> element_type_strs;
  for (const Type* element_type : tuple_type->element_types()) {
    element_type_strs.push_back(UnpackedMutableTypeString(*element_type));
  }
  return absl::StrFormat("xls::MutableTupleView<%s>",
                         absl::StrJoin(element_type_strs, ", "));
}

// Emits the code necessary to convert a u32/i32 value to its corresponding
// packed view.
std::string ConvertUint(std::string_view name, const Type& type) {
  CHECK(type.IsBits());

  return absl::StrFormat(
      "xls::PackedBitsView<%d> %s_view(absl::bit_cast<uint8_t*>(&%s), 0)",
      type.GetFlatBitCount(), name, name);
}

// Emits the code necessary to convert a float value to its corresponding
// packed view.
std::string ConvertFloat(std::string_view name) {
  return absl::StrCat(
      "xls::PackedTupleView<xls::PackedBitsView<1>, xls::PackedBitsView<8>, "
      "xls::PackedBitsView<23>> ",
      name, "_view(absl::bit_cast<uint8_t*>(&", name, "), 0)");
}

// Emits the code necessary to convert a double value to its corresponding
// packed view.
std::string ConvertDouble(std::string_view name) {
  return absl::StrCat(
      "xls::PackedTupleView<xls::PackedBitsView<1>, xls::PackedBitsView<11>, "
      "xls::PackedBitsView<52>> ",
      name, "_view(absl::bit_cast<uint8_t*>(&", name, "), 0)");
}

// Determines if the input type matches some other/simpler data type, and if so,
// returns it.
// Does not currently match > 1 specialization; i.e., if there were two types
// that could be specializations of a param.
std::optional<std::string> MatchTypeSpecialization(const Type& type) {
  // No need at present for anything fancy. Cascading if/else works.
  std::string type_string;
  if (MatchUint(type, &type_string)) {
    // Bits objects are an ordered of bits and have no notion of signedness so
    // they are best represented as unsigned integer data types in C/C++.
    return type_string;
  }
  if (MatchFloat(type)) {
    return "float";
  }
  if (MatchDouble(type)) {
    return "double";
  }

  return std::nullopt;
}

// Simple matching "driver" for emitting logic to convert a simple type into an
// XLS view.
// Pretty bare-bones at present, but will be expanded depending on need.
std::optional<std::string> CreateConversion(std::string_view name,
                                             const Type& type) {
  std::string type_string;
  if (MatchUint(type, &type_string)) {
    return ConvertUint(name, type);
  }
  if (MatchFloat(type)) {
    return ConvertFloat(name);
  }
  if (MatchDouble(type)) {
    return ConvertDouble(name);
  }

  return std::nullopt;
}

// Currently, we only support specialized interfaces if all the params and
// returns are specializable, as that's our current use case.
// To change this, we'd need to convert any non-specializable Values or
// non-packed views into packed views.
bool IsSpecializable(const Function& function) {
  auto [params, return_type] = GetSignature(function);

  for (const Param* param : params) {
    const Type& param_type = *param->GetType();
    if (!MatchTypeSpecialization(param_type).has_value()) {
      return false;
    }
  }

  return MatchTypeSpecialization(*return_type).has_value();
}

// Returns the specialized decl of the given function or an empty string, if not
// applicable.
std::string CreateDeclSpecialization(const Function& function,
                                     std::string prepend_class_name = "") {
  if (!IsSpecializable(function)) {
    return "";
  }

  // From here on, we know all elements are specializable.
  auto [params, return_type] = GetSignature(function);
  std::vector<std::string> param_strs;
  for (const Param* param : params) {
    const Type& param_type = *param->GetType();
    std::string specialization = MatchTypeSpecialization(param_type).value();
    param_strs.push_back(absl::StrCat(specialization, " ", param->name()));
  }

  std::string return_type_string =
      MatchTypeSpecialization(*return_type).value();

  if (!prepend_class_name.empty()) {
    absl::StrAppend(&prepend_class_name, "::");
  }

  return absl::StrFormat("absl::StatusOr<%s> %sRun(%s);", return_type_string,
                         prepend_class_name, absl::StrJoin(param_strs, ", "));
}

std::string CreateImplSpecialization(const Function& function,
                                     std::string_view class_name) {
  if (!IsSpecializable(function)) {
    return "";
  }

  // Get the decl, but remove the trailing semicolon.
  std::string signature =
      CreateDeclSpecialization(function, std::string(class_name));
  signature.pop_back();

  bool implicit_token_convention = false;
  auto [params, return_type] =
      GetSignature(function, &implicit_token_convention);

  std::vector<std::string> param_conversions;
  std::vector<std::string> param_names;

  if (implicit_token_convention) {
    param_conversions.push_back(
        "  uint8_t token = 0; xls::PackedBitsView<0> token_view(&token, 0)");
    param_conversions.push_back(
        "  uint8_t activated = 1; xls::PackedBitsView<1> "
        "activated_view(&activated, 0)");
    param_names.push_back("token_view");
    param_names.push_back("activated_view");
  }

  for (const Param* param : params) {
    // As with decls, we know conversions for all elements are possible, so we
    // can get values directly from the result std::optionals.
    std::string conversion =
        CreateConversion(param->name(), *param->GetType()).value();
    param_conversions.push_back(absl::StrCat("  ", conversion));
    param_names.push_back(absl::StrCat(param->name(), "_view"));
  }

  // Do the same for the return type - this requires allocating "buffer" space
  std::string return_spec = MatchTypeSpecialization(*return_type).value();
  param_conversions.push_back(absl::StrFormat(
      "  %s return_value;\n"
      "  %s",
      return_spec, CreateConversion("return_value", *return_type).value()));
  param_names.push_back("return_value_view");
  return absl::StrFormat(R"(%s {
%s;
  XLS_RETURN_IF_ERROR(jit_->RunWithPackedViews(%s));
  return return_value;
})",
                         signature, absl::StrJoin(param_conversions, ";\n"),
                         absl::StrJoin(param_names, ", "));
}

}  // namespace

static std::string GenerateWrapperHeader(
    const Function& function, std::string_view class_name,
    std::string_view wrapper_namespace,
    const std::filesystem::path& header_path,
    const std::filesystem::path& genfiles_path) {
  // Template substitution strings:
  //  {{class_name}} : Class name
  //  {{params}} : Function params
  //  {{function_name}} : Function name
  //  {{packed_params}} : Packed view params
  //  {{specialization}} : Any interfaces for specially-matched types, e.g., an
  //       interface that takes a float for a
  //       PackedTupleView<PackedBitsView<1>,...>.
  //  {{header_guard}} : Header guard.
  constexpr const char kHeaderTemplate[] =
      R"(// Automatically-generated file! DO NOT EDIT!
#ifndef {{header_guard}}
#define {{header_guard}}
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/jit/function_jit.h"
#include "xls/public/value.h"

namespace {{namespace}} {

// JIT execution wrapper for the {{function_name}} XLS IR module.
class {{class_name}} {
 public:
  static absl::StatusOr<std::unique_ptr<{{class_name}}>> Create();
  xls::FunctionJit* jit() { return jit_.get(); }

  absl::StatusOr<xls::Value> Run({{params}});
  absl::Status Run({{packed_params}});
  absl::Status Run({{unpacked_params}});
  {{specialization}}

 private:
  {{class_name}}(std::unique_ptr<xls::Package> package,
                 std::unique_ptr<xls::FunctionJit> jit);

  std::unique_ptr<xls::Package> package_;
  std::unique_ptr<xls::FunctionJit> jit_;
};

}  // namespace {{namespace}}

#endif  // {{header_guard}}
)";

  std::vector<std::string> param_strs;
  std::vector<std::string> packed_param_strs;
  std::vector<std::string> unpacked_param_strs;
  auto [params, return_type] = GetSignature(function);
  for (const Param* param : params) {
    param_strs.push_back(absl::StrCat("xls::Value ", param->name()));
    packed_param_strs.push_back(
        absl::StrCat(PackedTypeString(*param->GetType()), " ", param->name()));
    unpacked_param_strs.push_back(absl::StrCat(
        UnpackedTypeString(*param->GetType()), " ", param->name()));
  }
  packed_param_strs.push_back(
      absl::StrCat(PackedTypeString(*return_type), " result"));
  unpacked_param_strs.push_back(
      absl::StrCat(UnpackedMutableTypeString(*return_type), " result"));

  // Transform "blah/genfiles/xls/foo/bar.h" into "XLS_FOO_BAR_H_"
  std::string header_guard =
      std::string(header_path).substr(std::string(genfiles_path).size() + 1);
  // Only keep reasonable header-macro characters, everything else -> '_'
  std::transform(header_guard.begin(), header_guard.end(), header_guard.begin(),
                 [](char c) { return absl::ascii_isalnum(c) ? c : '_'; });
  header_guard = absl::StrCat(absl::AsciiStrToUpper(header_guard), "_");

  absl::flat_hash_map<std::string, std::string> substitution_map;
  substitution_map["{{class_name}}"] = class_name;
  substitution_map["{{namespace}}"] = wrapper_namespace;
  substitution_map["{{params}}"] = absl::StrJoin(param_strs, ", ");
  substitution_map["{{function_name}}"] = function.name();
  substitution_map["{{packed_params}}"] =
      absl::StrJoin(packed_param_strs, ", ");
  substitution_map["{{unpacked_params}}"] =
      absl::StrJoin(unpacked_param_strs, ", ");
  substitution_map["{{specialization}}"] = CreateDeclSpecialization(function);
  substitution_map["{{header_guard}}"] = header_guard;
  return absl::StrReplaceAll(kHeaderTemplate, substitution_map);
}

static std::string GenerateWrapperSource(
    const Function& function, std::string_view class_name,
    std::string_view wrapper_namespace,
    const std::filesystem::path& header_path) {
  // Use an extra '-' delimiter so we can embed a traditional-looking raw string
  // in the source. Template substitution strings:
  //  {{class_name}} : Class name
  //  {{ir_text}} : IR text
  //  {{params}} : Param list
  //  {{args}} : Arg list
  //  {{args_size}} : Arg list size
  //  {{header_path}} : Header path
  //  {{function_name}} : Function name (not camelized)
  //  {{run_params}} : Packed Run() params
  //  {{run_unpacked_params}} : Unpacked Run() params
  //  {{run_with_views_args}} : Packed RunWithPackedViews() arguments
  //  {{specialization}} : Specially-matched type implementations (if any)
  //  {{value_locals}}: "Value" routine locals.
  //  {{value_postprocessing}}: "Value" routine postprocessing.
  //  {{packed_locals}}: "Packed" routine locals.
  //  {{unpacked_locals}}: "Unpacked" routine locals.
  constexpr const char kSourceTemplate[] =
      R"-(// Automatically-generated file! DO NOT EDIT!
#include "{{header_path}}"
#include "xls/common/status/status_macros.h"
#include "xls/public/ir_parser.h"

namespace {{wrapper_namespace}} {

constexpr const char ir_text[] = R"original_ir({{ir_text}}
)original_ir";

absl::StatusOr<std::unique_ptr<{{class_name}}>> {{class_name}}::Create() {
  XLS_ASSIGN_OR_RETURN(auto package, xls::ParsePackage(ir_text, /*filename=*/std::nullopt));
  XLS_ASSIGN_OR_RETURN(xls::Function* function,
                       package->GetFunction("{{function_name}}"));
  XLS_ASSIGN_OR_RETURN(auto jit, xls::FunctionJit::Create(function));
  return absl::WrapUnique(new {{class_name}}(std::move(package), std::move(jit)));
}

{{class_name}}::{{class_name}}(std::unique_ptr<xls::Package> package,
                               std::unique_ptr<xls::FunctionJit> jit)
    : package_(std::move(package)), jit_(std::move(jit)) { }

absl::StatusOr<xls::Value> {{class_name}}::Run({{params}}) {
  {{value_locals}}
  xls::Value args[{{args_size}}] = { {{args}} };
  // Special form to handle zero-argument spans.
  XLS_ASSIGN_OR_RETURN(xls::Value _retval,
                       DropInterpreterEvents(jit_->Run(absl::MakeSpan(args, {{args_size}}))));
  {{value_postprocessing}}
  return _retval;
}

absl::Status {{class_name}}::Run({{run_params}}) {
  {{packed_locals}}
  return jit_->RunWithPackedViews({{run_with_views_args}});
}

absl::Status {{class_name}}::Run({{run_unpacked_params}}) {
  {{unpacked_locals}}
  return jit_->RunWithUnpackedViews({{run_with_views_args}});
}

{{specialization}}

}  // namespace {{wrapper_namespace}}
)-";
  std::vector<std::string> param_list;
  std::vector<std::string> packed_param_list;
  std::vector<std::string> unpacked_param_list;
  bool implicit_token_convention = false;
  auto [params, return_type] =
      GetSignature(function, &implicit_token_convention);
  for (const Param* param : params) {
    param_list.push_back(absl::StrCat("xls::Value ", param->name()));
    packed_param_list.push_back(
        absl::StrCat(PackedTypeString(*param->GetType()), " ", param->name()));
    unpacked_param_list.push_back(absl::StrCat(
        UnpackedTypeString(*param->GetType()), " ", param->name()));
  }
  packed_param_list.push_back(
      absl::StrCat(PackedTypeString(*return_type), " result"));
  unpacked_param_list.push_back(
      absl::StrCat(UnpackedMutableTypeString(*return_type), " result"));

  std::string params_str = absl::StrJoin(param_list, ", ");
  std::string packed_params_str = absl::StrJoin(packed_param_list, ", ");
  std::string unpacked_params_str = absl::StrJoin(unpacked_param_list, ", ");

  std::string value_locals;
  std::string packed_locals;
  std::string unpacked_locals;
  std::string retval_handling;
  std::vector<std::string> arg_list;
  if (implicit_token_convention) {
    arg_list.push_back("_token");
    arg_list.push_back("_activated");
    value_locals =
        "xls::Value _token = xls::Value::Token();\n"
        "  xls::Value _activated = xls::Value::Bool(true);";
    packed_locals =
        "uint8_t _token_value = 0; xls::PackedBitsView<0> "
        "_token(&_token_value, "
        "0);\n"
        "  uint8_t _activated_value = 1; xls::PackedBitsView<1> "
        "_activated(&_activated_value, 1);";
    unpacked_locals =
        "uint8_t _token_value = 0; xls::BitsView<0> "
        "_token(&_token_value, "
        "0);\n"
        "  uint8_t _activated_value = 1; xls::BitsView<1> "
        "_activated(&_activated_value, 1);";
    retval_handling = "_retval = _retval.elements()[1];";
  }
  for (const Param* param : params) {
    arg_list.push_back(std::string(param->name()));
  }
  int num_unpacked_args = arg_list.size();
  std::string unpacked_args = absl::StrJoin(arg_list, ", ");
  arg_list.push_back("result");
  std::string run_with_views_args = absl::StrJoin(arg_list, ", ");

  std::string specialization = CreateImplSpecialization(function, class_name);
  absl::flat_hash_map<std::string, std::string> substitution_map;
  substitution_map["{{class_name}}"] = class_name;
  substitution_map["{{ir_text}}"] = function.package()->DumpIr();
  substitution_map["{{params}}"] = params_str;
  substitution_map["{{args}}"] = unpacked_args;
  substitution_map["{{args_size}}"] = absl::StrCat(num_unpacked_args);
  substitution_map["{{header_path}}"] = header_path.string();
  substitution_map["{{function_name}}"] = function.name();
  substitution_map["{{run_params}}"] = packed_params_str;
  substitution_map["{{run_unpacked_params}}"] = unpacked_params_str;
  substitution_map["{{run_with_views_args}}"] = run_with_views_args;
  substitution_map["{{specialization}}"] = specialization;
  substitution_map["{{value_locals}}"] = value_locals;
  substitution_map["{{value_postprocessing}}"] = retval_handling;
  substitution_map["{{packed_locals}}"] = packed_locals;
  substitution_map["{{unpacked_locals}}"] = packed_locals;
  substitution_map["{{wrapper_namespace}}"] = wrapper_namespace;
  return absl::StrReplaceAll(kSourceTemplate, substitution_map);
}

GeneratedJitWrapper GenerateJitWrapper(
    const Function& function, std::string_view class_name,
    std::string_view wrapper_namespace,
    const std::filesystem::path& header_path,
    const std::filesystem::path& genfiles_path) {
  GeneratedJitWrapper wrapper;
  wrapper.header = GenerateWrapperHeader(
      function, class_name, wrapper_namespace, header_path, genfiles_path);
  wrapper.source = GenerateWrapperSource(function, class_name,
                                         wrapper_namespace, header_path);
  return wrapper;
}

}  // namespace xls
