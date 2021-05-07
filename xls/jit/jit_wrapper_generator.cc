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

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"

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
    XLS_CHECK(return_type->IsTuple() &&
              return_type->AsTupleOrDie()->size() == 2)
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
  } else if (bit_count <= 16) {
    *enclosing_type = "uint16_t";
    return true;
  } else if (bit_count <= 32) {
    *enclosing_type = "uint32_t";
    return true;
  } else if (bit_count <= 64) {
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
    return absl::StrCat("PackedBitsView<", type.GetFlatBitCount(), ">");
  } else if (type.IsArray()) {
    const ArrayType* array_type = type.AsArrayOrDie();
    std::string element_type_str =
        PackedTypeString(*array_type->element_type());
    return absl::StrFormat("PackedArrayView<%s, %d>", element_type_str,
                           array_type->size());
  } else {
    // Is tuple!
    XLS_CHECK(type.IsTuple()) << type.ToString();
    const TupleType* tuple_type = type.AsTupleOrDie();
    std::vector<std::string> element_type_strs;
    for (const Type* element_type : tuple_type->element_types()) {
      element_type_strs.push_back(PackedTypeString(*element_type));
    }
    return absl::StrFormat("PackedTupleView<%s>",
                           absl::StrJoin(element_type_strs, ", "));
  }
}

// Emits the code necessary to convert a u32/i32 value to its corresponding
// packed view.
std::string ConvertUint(absl::string_view name, const Type& type) {
  XLS_CHECK(type.IsBits());

  return absl::StrFormat(
      "PackedBitsView<%d> %s_view(absl::bit_cast<uint8_t*>(&%s), 0)",
      type.GetFlatBitCount(), name, name);
}

// Emits the code necessary to convert a float value to its corresponding
// packed view.
std::string ConvertFloat(absl::string_view name) {
  return absl::StrCat(
      "PackedTupleView<PackedBitsView<1>, PackedBitsView<8>, "
      "PackedBitsView<23>> ",
      name, "_view(absl::bit_cast<uint8_t*>(&", name, "), 0)");
}

// Emits the code necessary to convert a double value to its corresponding
// packed view.
std::string ConvertDouble(absl::string_view name) {
  return absl::StrCat(
      "PackedTupleView<PackedBitsView<1>, PackedBitsView<11>, "
      "PackedBitsView<52>> ",
      name, "_view(absl::bit_cast<uint8_t*>(&", name, "), 0)");
}

// Determines if the input type matches some other/simpler data type, and if so,
// returns it.
// Does not currently match > 1 specialization; i.e., if there were two types
// that could be specializations of a param.
absl::optional<std::string> MatchTypeSpecialization(const Type& type) {
  // No need at present for anything fancy. Cascading if/else works.
  std::string type_string;
  if (MatchUint(type, &type_string)) {
    // Bits objects are an ordered of bits and have no notion of signedness so
    // they are best represented as unsigned integer data types in C/C++.
    return type_string;
  } else if (MatchFloat(type)) {
    return "float";
  } else if (MatchDouble(type)) {
    return "double";
  }

  return absl::nullopt;
}

// Simple matching "driver" for emitting logic to convert a simple type into an
// XLS view.
// Pretty bare-bones at present, but will be expanded depending on need.
absl::optional<std::string> CreateConversion(absl::string_view name,
                                             const Type& type) {
  std::string type_string;
  if (MatchUint(type, &type_string)) {
    return ConvertUint(name, type);
  } else if (MatchFloat(type)) {
    return ConvertFloat(name);
  } else if (MatchDouble(type)) {
    return ConvertDouble(name);
  }

  return absl::nullopt;
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
                                     absl::string_view class_name) {
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
        "  uint8_t token = 0; PackedBitsView<0> token_view(&token, 0)");
    param_conversions.push_back(
        "  uint8_t activated = 1; PackedBitsView<1> activated_view(&activated, "
        "0)");
    param_names.push_back("token_view");
    param_names.push_back("activated_view");
  }

  for (const Param* param : params) {
    // As with decls, we know conversions for all elements are possible, so we
    // can get values directly from the result absl::optionals.
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

std::string GenerateWrapperHeader(const Function& function,
                                  absl::string_view class_name,
                                  const std::filesystem::path& header_path,
                                  const std::filesystem::path& genfiles_path) {
  // $0 : Class name
  // $1 : Function params
  // $2 : Function name
  // $3 : Packed view params
  // $4 : Any interfaces for specially-matched types, e.g., an interface that
  //      takes a float for a PackedTupleView<PackedBitsView<1>, ...>.
  // $5 : Header guard.
  constexpr const char header_template[] =
      R"(// Automatically-generated file! DO NOT EDIT!
#ifndef $5
#define $5
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/jit/ir_jit.h"
#include "xls/public/value.h"

namespace xls {

// JIT execution wrapper for the $2 XLS IR module.
class $0 {
 public:
  static absl::StatusOr<std::unique_ptr<$0>> Create();
  IrJit* jit() { return jit_.get(); }

  absl::StatusOr<Value> Run($1);
  absl::Status Run($3);
  $4

 private:
  $0(std::unique_ptr<Package> package, std::unique_ptr<IrJit> jit);

  std::unique_ptr<Package> package_;
  std::unique_ptr<IrJit> jit_;
};

}  // namespace xls

#endif  // $5
)";

  std::vector<std::string> param_strs;
  std::vector<std::string> packed_param_strs;
  auto [params, return_type] = GetSignature(function);
  for (const Param* param : params) {
    param_strs.push_back(absl::StrCat("Value ", param->name()));
    packed_param_strs.push_back(
        absl::StrCat(PackedTypeString(*param->GetType()), " ", param->name()));
  }
  packed_param_strs.push_back(
      absl::StrCat(PackedTypeString(*return_type), " result"));

  // Transform "blah/genfiles/xls/foo/bar.h" into "XLS_FOO_BAR_H_"
  std::string header_guard =
      std::string(header_path).substr(std::string(genfiles_path).size() + 1);
  header_guard = absl::StrReplaceAll(
      header_guard,
      {
          {absl::StrFormat("%c", header_path.preferred_separator), "_"},
          {".", "_"},
      });
  header_guard = absl::StrCat(absl::AsciiStrToUpper(header_guard), "_");

  return absl::Substitute(header_template, class_name,
                          absl::StrJoin(param_strs, ", "), function.name(),
                          absl::StrJoin(packed_param_strs, ", "),
                          CreateDeclSpecialization(function), header_guard);
}

std::string GenerateWrapperSource(const Function& function,
                                  absl::string_view class_name,
                                  const std::filesystem::path& header_path) {
  // Use an extra '-' delimiter so we can embed a traditional-looking raw string
  // in the source.
  //  $0 : Class name
  //  $1 : IR text
  //  $2 : Param list
  //  $3 : Arg list
  //  $4 : Arg list size
  //  $5 : Header path
  //  $6 : Function name (not camelized)
  //  $7 : Packed Run() params
  //  $8 : Packed RunWithPackedViews() arguments
  //  $9 : Specially-matched type implementations (if any)
  //
  // Note that Substitute() only supports up to $9, so we need a second pass
  // that uses escaped values to exceed that number.
  //
  //  $$0: "Value" routine locals.
  //  $$1: "Value" routine postprocessing.
  //  $$2: "Packed" routine locals.
  constexpr const char source_template[] =
      R"-(// Automatically-generated file! DO NOT EDIT!
#include "$5"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"

namespace xls {

constexpr const char ir_text[] = R"($1
)";

absl::StatusOr<std::unique_ptr<$0>> $0::Create() {
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(Function* function, package->GetFunction("$6"));
  XLS_ASSIGN_OR_RETURN(auto jit, IrJit::Create(function));
  return absl::WrapUnique(new $0(std::move(package), std::move(jit)));
}

$0::$0(std::unique_ptr<Package> package, std::unique_ptr<IrJit> jit)
    : package_(std::move(package)), jit_(std::move(jit)) { }

absl::StatusOr<Value> $0::Run($2) {
  $$0
  Value args[$4] = { $3 };
  // Special form to handle zero-argument spans.
  XLS_ASSIGN_OR_RETURN(Value _retval, jit_->Run(absl::MakeSpan(args, $4)));
  $$1
  return _retval;
}

absl::Status $0::Run($7) {
  $$2
  return jit_->RunWithPackedViews($8);
}

$9

}  // namespace xls
)-";
  std::vector<std::string> param_list;
  std::vector<std::string> packed_param_list;
  bool implicit_token_convention = false;
  auto [params, return_type] =
      GetSignature(function, &implicit_token_convention);
  for (const Param* param : params) {
    param_list.push_back(absl::StrCat("Value ", param->name()));
    packed_param_list.push_back(
        absl::StrCat(PackedTypeString(*param->GetType()), " ", param->name()));
  }
  packed_param_list.push_back(
      absl::StrCat(PackedTypeString(*return_type), " result"));

  std::string params_str = absl::StrJoin(param_list, ", ");
  std::string packed_params_str = absl::StrJoin(packed_param_list, ", ");

  std::string value_locals;
  std::string packed_locals;
  std::string retval_handling;
  std::vector<std::string> arg_list;
  if (implicit_token_convention) {
    arg_list.push_back("_token");
    arg_list.push_back("_activated");
    value_locals =
        "Value _token = Value::Token();\n"
        "  Value _activated = Value::Bool(true);";
    packed_locals =
        "uint8_t _token_value = 0; PackedBitsView<0> _token(&_token_value, "
        "0);\n"
        "  uint8_t _activated_value = 1; PackedBitsView<1> "
        "_activated(&_activated_value, 1);";
    retval_handling = "_retval = _retval.elements()[1];";
  }
  for (const Param* param : params) {
    arg_list.push_back(std::string(param->name()));
  }
  int num_unpacked_args = arg_list.size();
  std::string unpacked_args = absl::StrJoin(arg_list, ", ");
  arg_list.push_back("result");
  std::string packed_args = absl::StrJoin(arg_list, ", ");

  std::string specialization = CreateImplSpecialization(function, class_name);

  std::string substituted = absl::Substitute(
      source_template, class_name, function.package()->DumpIr(), params_str,
      unpacked_args, num_unpacked_args, header_path.string(), function.name(),
      packed_params_str, packed_args, specialization);
  return absl::Substitute(substituted, value_locals, retval_handling,
                          packed_locals);
}

GeneratedJitWrapper GenerateJitWrapper(
    const Function& function, const std::string& class_name,
    const std::filesystem::path& header_path,
    const std::filesystem::path& genfiles_path) {
  GeneratedJitWrapper wrapper;
  wrapper.header =
      GenerateWrapperHeader(function, class_name, header_path, genfiles_path);
  wrapper.source = GenerateWrapperSource(function, class_name, header_path);
  return wrapper;
}

}  // namespace xls
