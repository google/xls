// Copyright 2020 Google LLC
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
#include "xls/ir/jit_wrapper_generator.h"

#include "absl/strings/substitute.h"

namespace xls {
namespace {

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
    const TupleType* tuple_type = type.AsTupleOrDie();
    std::vector<std::string> element_type_strs;
    for (const Type* element_type : tuple_type->element_types()) {
      element_type_strs.push_back(PackedTypeString(*element_type));
    }
    return absl::StrFormat("PackedTupleView<%s>",
                           absl::StrJoin(element_type_strs, ", "));
  }
}

// Returns true if the given type matches the C float type layout.
bool MatchFloat(const Type& type) {
  if (!type.IsTuple()) {
    return false;
  }

  const TupleType* tuple_type = type.AsTupleOrDie();
  auto element_types = tuple_type->element_types();
  if (element_types[0]->IsBits() && element_types[0]->GetFlatBitCount() == 23 &&
      element_types[1]->IsBits() && element_types[1]->GetFlatBitCount() == 8 &&
      element_types[2]->IsBits() && element_types[2]->GetFlatBitCount() == 1) {
    return true;
  }
  return false;
}

// Emits the code necessary to convert a float value to its corresponding
// packed view.
std::string ConvertFloat(const std::string& name) {
  return absl::StrCat(
      "PackedTupleView<PackedBitsView<23>, PackedBitsView<8>, "
      "PackedBitsView<1>> ",
      name, "_view(reinterpret_cast<uint8*>(&", name, "), 0)");
}

// Determines if the input type matches some other/simpler data type, and if so,
// returns it.
// Does not currently match > 1 specialization; i.e., if there were two types
// that could be specializations of a param.
absl::optional<std::string> MatchTypeSpecialization(const Type& type) {
  // No need at present for anything fancy. Cascading if/else works.
  if (MatchFloat(type)) {
    return "float";
  }

  return absl::nullopt;
}

// Simple matching "driver" for emitting logic to convert a simple type into an
// XLS view.
// Pretty bare-bones at present, but will be expanded depending on need.
absl::optional<std::string> CreateConversion(const std::string& name,
                                             const Type& type) {
  if (MatchFloat(type)) {
    return ConvertFloat(name);
  }

  return absl::nullopt;
}

// Currently, we only support specialized interfaces if all the params and
// returns are specializable, as that's our current use case.
// To change this, we'd need to convert any non-specializable Values or
// non-packed views into packed views.
bool IsSpecializable(const Function& function) {
  for (const Param* param : function.params()) {
    const Type& param_type = *param->GetType();
    if (!MatchTypeSpecialization(param_type).has_value()) {
      return false;
    }
  }

  const Type& return_type = *function.return_value()->GetType();
  return MatchTypeSpecialization(return_type).has_value();
}

// Returns the specialized decl of the given function or an empty string, if not
// applicable.
std::string CreateDeclSpecialization(const Function& function,
                                     std::string prepend_class_name = "") {
  if (!IsSpecializable(function)) {
    return "";
  }

  // From here on, we know all elements are specializable, so we can directly
  // the values out of the absl::optionals.
  std::vector<std::string> params;
  for (const Param* param : function.params()) {
    const Type& param_type = *param->GetType();
    std::string specialization = MatchTypeSpecialization(param_type).value();
    params.push_back(absl::StrCat(specialization, " ", param->name()));
  }

  const Type& return_type = *function.return_value()->GetType();
  std::string return_type_string = MatchTypeSpecialization(return_type).value();

  if (!prepend_class_name.empty()) {
    absl::StrAppend(&prepend_class_name, "::");
  }

  return absl::StrFormat("xabsl::StatusOr<%s> %sRun(%s);", return_type_string,
                         prepend_class_name, absl::StrJoin(params, ", "));
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

  // Convert all "simple" types to their XLS equivalents.
  std::vector<std::string> param_conversions;
  std::vector<std::string> param_names;
  for (const Param* param : function.params()) {
    // As with decls, we know conversions for all elements are possible, so we
    // can get values directly from the result absl::optionals.
    std::string conversion =
        CreateConversion(param->name(), *param->GetType()).value();
    param_conversions.push_back(absl::StrCat("  ", conversion));
    param_names.push_back(absl::StrCat(param->name(), "_view"));
  }

  // Do the same for the return type - this requires allocating "buffer" space
  const Type& return_type = *function.return_value()->GetType();
  std::string return_spec = MatchTypeSpecialization(return_type).value();
  param_conversions.push_back(
      absl::StrFormat("  %s return_value;\n"
                      "#if __has_feature(memory_sanitizer)\n"
                      "  __msan_unpoison(&return_value, sizeof(%s));\n"
                      "#endif\n"
                      "  %s",
                      return_spec, return_spec,
                      CreateConversion("return_value", return_type).value()));
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
                                  absl::string_view class_name) {
  // $0 : Class name
  // $1 : Function params
  // $2 : Function name
  // $3 : Packed view params
  // $4 : Any interfaces for specially-matched types, e.g., an interface that
  //      takes a float for a PackedTupleView<PackedBitsView<23>, ...>.
  constexpr const char header_template[] =
      R"(// Automatically-generated file! DO NOT EDIT!
#include <memory>

#include "absl/status/status.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/llvm_ir_jit.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"

namespace xls {

// JIT execution wrapper for the $2 XLS IR module.
class $0 {
 public:
  static xabsl::StatusOr<std::unique_ptr<$0>> Create();
  LlvmIrJit* jit() { return jit_.get(); }

  xabsl::StatusOr<Value> Run($1);
  absl::Status Run($3);
  $4

 private:
  $0(std::unique_ptr<Package> package, std::unique_ptr<LlvmIrJit> jit);

  std::unique_ptr<Package> package_;
  std::unique_ptr<LlvmIrJit> jit_;
};

}  // namespace xls
)";

  std::vector<std::string> params;
  std::vector<std::string> packed_params;
  for (const Param* param : function.params()) {
    params.push_back(absl::StrCat("Value ", param->name()));
    packed_params.push_back(
        absl::StrCat(PackedTypeString(*param->GetType()), " ", param->name()));
  }
  packed_params.push_back(absl::StrCat(
      PackedTypeString(*function.return_value()->GetType()), " result"));

  return absl::Substitute(
      header_template, class_name, absl::StrJoin(params, ", "), function.name(),
      absl::StrJoin(packed_params, ", "), CreateDeclSpecialization(function));
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
  constexpr const char source_template[] =
      R"-(// Automatically-generated file! DO NOT EDIT!
#include "$5"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"

namespace xls {

constexpr const char ir_text[] = R"($1
)";

xabsl::StatusOr<std::unique_ptr<$0>> $0::Create() {
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(Function* function, package->GetFunction("$6"));
  XLS_ASSIGN_OR_RETURN(auto jit, LlvmIrJit::Create(function));
  return absl::WrapUnique(new $0(std::move(package), std::move(jit)));
}

$0::$0(std::unique_ptr<Package> package, std::unique_ptr<LlvmIrJit> jit)
    : package_(std::move(package)), jit_(std::move(jit)) { }

xabsl::StatusOr<Value> $0::Run($2) {
  Value args[$4] = { $3 };
  // Special form to handle zero-argument spans.
  return jit_->Run(absl::MakeSpan(args, $4));
}

absl::Status $0::Run($7) {
  return jit_->RunWithPackedViews($8);
}

$9

}  // namespace xls
)-";
  std::vector<std::string> param_list;
  std::vector<std::string> packed_param_list;
  for (const Param* param : function.params()) {
    param_list.push_back(absl::StrCat("Value ", param->name()));
    packed_param_list.push_back(
        absl::StrCat(PackedTypeString(*param->GetType()), " ", param->name()));
  }
  packed_param_list.push_back(absl::StrCat(
      PackedTypeString(*function.return_value()->GetType()), " result"));

  std::string params = absl::StrJoin(param_list, ", ");
  std::string packed_params = absl::StrJoin(packed_param_list, ", ");

  std::vector<std::string> arg_list;
  for (const Param* param : function.params()) {
    arg_list.push_back(param->name());
  }
  int num_unpacked_args = arg_list.size();
  std::string unpacked_args = absl::StrJoin(arg_list, ", ");
  arg_list.push_back("result");
  std::string packed_args = absl::StrJoin(arg_list, ", ");

  std::string specialization = CreateImplSpecialization(function, class_name);

  return absl::Substitute(
      source_template, class_name, function.package()->DumpIr(), params,
      unpacked_args, num_unpacked_args, header_path.string(), function.name(),
      packed_params, packed_args, specialization);
}

GeneratedJitWrapper GenerateJitWrapper(
    const Function& function, const std::string& class_name,
    const std::filesystem::path& header_path) {
  GeneratedJitWrapper wrapper;
  wrapper.header = GenerateWrapperHeader(function, class_name);
  wrapper.source = GenerateWrapperSource(function, class_name, header_path);
  return wrapper;
}

}  // namespace xls
