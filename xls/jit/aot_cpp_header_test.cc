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

#include "xls/jit/aot_cpp_header.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/jit/aot_entrypoint.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

constexpr std::string_view kNamespace = "aot_example::generated";
constexpr std::string_view kEntrypoint = "Transform";
constexpr std::string_view kPackedSymbol = "__test_package_transform_packed";

TypeProto Bits(int64_t bit_count) {
  TypeProto type;
  type.set_type_enum(TypeProto::BITS);
  type.set_bit_count(bit_count);
  return type;
}

TypeProto Array(const TypeProto& element, int64_t size) {
  TypeProto type;
  type.set_type_enum(TypeProto::ARRAY);
  type.set_array_size(size);
  *type.mutable_array_element() = element;
  return type;
}

TypeProto Tuple(std::initializer_list<TypeProto> elements) {
  TypeProto type;
  type.set_type_enum(TypeProto::TUPLE);
  for (const TypeProto& element : elements) {
    *type.add_tuple_elements() = element;
  }
  return type;
}

int64_t FlatBitCount(const TypeProto& type) {
  switch (type.type_enum()) {
    case TypeProto::BITS:
      return type.bit_count();
    case TypeProto::ARRAY:
      return FlatBitCount(type.array_element()) * type.array_size();
    case TypeProto::TUPLE: {
      int64_t total = 0;
      for (const TypeProto& element : type.tuple_elements()) {
        total += FlatBitCount(element);
      }
      return total;
    }
    default:
      return 0;
  }
}

int64_t PackedBits(int64_t flat_bit_count) {
  if (flat_bit_count == 0) {
    return 8;
  }
  return ((flat_bit_count + 7) / 8) * 8;
}

// Fills a function entrypoint with the given structural parameters and result.
AotPackageEntrypointsProto MakeFunctionPackage(
    const std::vector<std::pair<std::string, TypeProto>>& parameters,
    const TypeProto& result) {
  AotPackageEntrypointsProto package;
  AotEntrypointProto* entrypoint = package.add_entrypoint();
  entrypoint->set_type(AotEntrypointProto::FUNCTION);
  entrypoint->set_xls_package_name("test_package");
  entrypoint->set_xls_function_identifier("test_fn");
  entrypoint->set_function_symbol("__pkg_test_fn");
  entrypoint->set_packed_function_symbol(std::string(kPackedSymbol));
  entrypoint->set_temp_buffer_size(64);
  entrypoint->set_temp_buffer_alignment(16);

  AotEntrypointProto::FunctionMetadataProto* function_metadata =
      entrypoint->mutable_function_metadata();
  PackageInterfaceProto::Function* function_interface =
      function_metadata->mutable_function_interface();
  for (const auto& [name, type] : parameters) {
    PackageInterfaceProto::NamedValue* param =
        function_interface->add_parameters();
    param->set_name(name);
    *param->mutable_type() = type;
    entrypoint->add_inputs_names(name);
    entrypoint->add_packed_input_buffer_sizes(PackedBits(FlatBitCount(type)));
  }
  *function_interface->mutable_result_type() = result;
  entrypoint->add_outputs_names("result");
  entrypoint->add_packed_output_buffer_sizes(PackedBits(FlatBitCount(result)));
  return package;
}

absl::StatusOr<std::string> Generate(
    const AotPackageEntrypointsProto& package,
    std::string_view cpp_namespace = kNamespace,
    std::string_view entrypoint_name = kEntrypoint) {
  return GenerateAotCppHeader(
      package,
      AotCppHeaderOptions{.cpp_namespace = std::string(cpp_namespace),
                          .entrypoint_name = std::string(entrypoint_name)});
}

class AotCppHeaderTest : public ::testing::Test {
 protected:
  AotPackageEntrypointsProto package_ =
      MakeFunctionPackage({{"state", Tuple({Bits(16), Bits(5), Bits(3)})},
                           {"ctx_in", Array(Bits(8), 8)},
                           {"input_byte", Bits(8)}},
                          Tuple({Bits(4), Array(Bits(1), 4)}));
};

TEST_F(AotCppHeaderTest, FullFunctionHeaderAndDeterminism) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package_));
  // Determinism.
  XLS_ASSERT_OK_AND_ASSIGN(std::string header_again, Generate(package_));
  EXPECT_EQ(header, header_again);

  // Header prelude and include guard computed from stable options.
  EXPECT_THAT(header,
              HasSubstr("Generated by XLS aot_compiler_main; do not edit."));
  EXPECT_THAT(
      header,
      HasSubstr(
          "#ifndef XLS_AOT_11_aot_example_9_generated_9_Transform_H_"));
  EXPECT_THAT(header, HasSubstr("namespace aot_example {"));
  EXPECT_THAT(header, HasSubstr("namespace generated {"));
  EXPECT_THAT(header, HasSubstr("namespace Transform {"));

  // Standard requires: no XLS/protobuf/absl includes, <tuple> present.
  EXPECT_THAT(header, HasSubstr("#include <array>"));
  EXPECT_THAT(header, HasSubstr("#include <cstddef>"));
  EXPECT_THAT(header, HasSubstr("#include <cstdint>"));
  EXPECT_THAT(header, HasSubstr("#include <tuple>"));

  // Support templates.
  EXPECT_THAT(header, HasSubstr("template <std::size_t Width>\nstruct Bits {"));
  EXPECT_THAT(
      header,
      HasSubstr(
          "template <typename ElementType, std::size_t Size>\nstruct Array {"));
  EXPECT_THAT(header,
              HasSubstr("template <typename... ElementTypes>\nstruct Tuple {"));
  EXPECT_THAT(
      header,
      HasSubstr("std::tuple_element_t<Index, std::tuple<ElementTypes...>>"));

  // ABI constants.
  EXPECT_THAT(header, HasSubstr("kInputCount = 3"));
  EXPECT_THAT(header, HasSubstr("kOutputCount = 1"));
  EXPECT_THAT(header, HasSubstr("kTemporaryBufferSize = 64"));
  EXPECT_THAT(header, HasSubstr("kTemporaryBufferAlignment = 16"));

  // Type aliases (indexed and semantic).
  EXPECT_THAT(header,
              HasSubstr("using Input0 = Tuple<Bits<16>, Bits<5>, Bits<3>>;"));
  EXPECT_THAT(header, HasSubstr("using Input1 = Array<Bits<8>, 8>;"));
  EXPECT_THAT(header, HasSubstr("using Input2 = Bits<8>;"));
  EXPECT_THAT(header, HasSubstr("using State = Input0;"));
  EXPECT_THAT(header, HasSubstr("using CtxIn = Input1;"));
  EXPECT_THAT(header, HasSubstr("using InputByte = Input2;"));
  EXPECT_THAT(header,
              HasSubstr("using Result = Tuple<Bits<4>, Array<Bits<1>, 4>>;"));

  // The packed symbol appears only in the asm label.
  EXPECT_THAT(header, HasSubstr(absl::StrCat("asm(\"", kPackedSymbol, "\")")));

  // Packed sizes and the compile-time checks.
  EXPECT_THAT(header,
              HasSubstr("kPackedInputBitCounts = {24, 64, 8}"));
  EXPECT_THAT(header, HasSubstr("kPackedOutputBitCounts = {8}"));
  EXPECT_THAT(
      header,
      HasSubstr("static_assert(PackedBitCount<Result::kBitCount> == "
                "kPackedOutputBitCounts[0], \"Result type mismatch with "
                "metadata.\");"));
}

TEST_F(AotCppHeaderTest, GoldenSimpleFunction) {
  AotPackageEntrypointsProto package = MakeFunctionPackage(
      {{"x", Bits(3)}, {"a", Array(Bits(8), 2)}}, Tuple({Bits(1), Bits(7)}));
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package));
  EXPECT_THAT(
      header,
      HasSubstr(
          "#ifndef XLS_AOT_11_aot_example_9_generated_9_Transform_H_"));
  EXPECT_THAT(header, HasSubstr("using Input0 = Bits<3>;"));
  EXPECT_THAT(header, HasSubstr("using X = Input0;"));
  EXPECT_THAT(header, HasSubstr("using Input1 = Array<Bits<8>, 2>;"));
  EXPECT_THAT(header, HasSubstr("using A = Input1;"));
  EXPECT_THAT(header, HasSubstr("using Result = Tuple<Bits<1>, Bits<7>>;"));
  EXPECT_THAT(header, HasSubstr("kPackedInputBitCounts = {8, 16}"));
  EXPECT_THAT(header, HasSubstr("kPackedOutputBitCounts = {8}"));
  // Self-contained: no non-std includes.
  EXPECT_THAT(header, Not(HasSubstr("#include \"")));
  EXPECT_THAT(header, Not(HasSubstr("#include <absl/")));
  EXPECT_THAT(header, Not(HasSubstr("#include <xls/")));
  EXPECT_THAT(header, Not(HasSubstr("#include <google/")));
}

TEST_F(AotCppHeaderTest, IndexedAliasesAlwaysPresent) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"input 1!", Bits(3)}, {"", Bits(4)}}, Bits(1));
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package));
  EXPECT_THAT(header, HasSubstr("using Input0 = Bits<3>;"));
  EXPECT_THAT(header, HasSubstr("using Input1 = Bits<4>;"));
  // Semantic aliases for those impossible names are skipped silently.
  EXPECT_THAT(header, Not(HasSubstr("using Input0 = Input0;")));
}

TEST_F(AotCppHeaderTest, SemanticAliasesAvoidGeneratedNameCollisions) {
  AotPackageEntrypointsProto package = MakeFunctionPackage(
      {{"transform_invoke_packed", Bits(8)}, {"same_name", Bits(8)},
       {"same__name", Bits(8)}},
      Bits(8));
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package));
  EXPECT_THAT(header, Not(HasSubstr("using TransformInvokePacked =")));
  EXPECT_THAT(header, HasSubstr("using SameName = Input1;"));
  EXPECT_THAT(header, Not(HasSubstr("using SameName = Input2;")));
}

TEST_F(AotCppHeaderTest, ZeroArgumentsUseStandardArray) {
  AotPackageEntrypointsProto package = MakeFunctionPackage({}, Bits(8));
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package));
  EXPECT_THAT(
      header,
      HasSubstr("std::array<std::size_t, kInputCount> "
                "kPackedInputBitCounts = {};"));
  EXPECT_THAT(header, Not(HasSubstr("kPackedInputBitCounts[0]")));
}

TEST_F(AotCppHeaderTest, PackedFunctionTypeAndSymbol) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package_));
  EXPECT_THAT(header, HasSubstr("using PackedFunction = std::int64_t (*)("));
  EXPECT_THAT(header, HasSubstr("const std::uint8_t* const* inputs"));
  EXPECT_THAT(
      header,
      HasSubstr(
          "extern \"C\" std::int64_t TransformInvokePacked("));
  EXPECT_THAT(header,
              HasSubstr("inline constexpr PackedFunction kPackedFunction = "
                        "&TransformInvokePacked;"));
  EXPECT_THAT(header, Not(HasSubstr("std::int64_t InvokePacked(")));
}

TEST_F(AotCppHeaderTest, EmptyTupleSupported) {
  // An empty tuple is structurally legal in the templates.
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"empty", Tuple({})}}, Bits(1));
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package));
  EXPECT_THAT(header, HasSubstr("using Input0 = Tuple<>;"));
}

TEST_F(AotCppHeaderTest, DeeplyNestedTypes) {
  AotPackageEntrypointsProto package = MakeFunctionPackage(
      {{"deep", Array(Tuple({Bits(2), Array(Tuple({Bits(3)}), 5)}), 4)}},
      Tuple({Bits(1)}));
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package));
  EXPECT_THAT(header, HasSubstr("using Input0 = Array<Tuple<Bits<2>, "
                                "Array<Tuple<Bits<3>>, 5>>, 4>;"));
}

TEST_F(AotCppHeaderTest, SymbolEscaping) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(8)}}, Bits(1));
  package.mutable_entrypoint(0)->set_packed_function_symbol("sym\"\\\n");
  XLS_ASSERT_OK_AND_ASSIGN(std::string header, Generate(package));
  EXPECT_THAT(header, HasSubstr("asm(\"sym\\\"\\\\\\n\")"));
}

TEST_F(AotCppHeaderTest, EmptyNamespaceIsAllowed) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string header,
                           Generate(package_, /*cpp_namespace=*/""));
  EXPECT_THAT(header, HasSubstr("#ifndef XLS_AOT_9_Transform_H_"));
  EXPECT_THAT(header, HasSubstr("namespace Transform {"));
}

TEST_F(AotCppHeaderTest, InvalidNamespaceErrors) {
  EXPECT_THAT(Generate(package_, "valid::an identifier::a"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("C++ identifier")));
  EXPECT_THAT(Generate(package_, "5leading_digit"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(AotCppHeaderTest, InvalidEntrypointNameErrors) {
  EXPECT_THAT(Generate(package_, kNamespace, "not valid"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Generate(package_, kNamespace, "class"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("C++ keyword")));
  EXPECT_THAT(Generate(package_, kNamespace, "__reserved"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Generate(package_, kNamespace, "_Reserved"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Generate(package_, kNamespace, "_private"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("reserved to the C++ implementation")));
  EXPECT_THAT(Generate(package_, "_private", kEntrypoint),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(AotCppHeaderTest, NonFunctionEntrypointErrors) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(3)}}, Bits(1));
  package.mutable_entrypoint(0)->set_type(AotEntrypointProto::PROC);
  EXPECT_THAT(Generate(package),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("FUNCTION entrypoints only")));

  package.mutable_entrypoint(0)->set_type(AotEntrypointProto::BLOCK);
  EXPECT_THAT(Generate(package),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("FUNCTION entrypoints only")));
}

TEST_F(AotCppHeaderTest, MultipleEntrypointsError) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(8)}}, Bits(8));
  package.add_entrypoint()->CopyFrom(package.entrypoint(0));
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("exactly one entrypoint")));
}

TEST_F(AotCppHeaderTest, MissingPackedSymbolErrors) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(8)}}, Bits(8));
  package.mutable_entrypoint(0)->clear_packed_function_symbol();
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("packed function symbol")));
  package.mutable_entrypoint(0)->set_packed_function_symbol("");
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("packed function symbol")));
}

TEST_F(AotCppHeaderTest, MissingOrNegativeTemporaryBufferMetadataErrors) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(8)}}, Bits(8));
  package.mutable_entrypoint(0)->clear_temp_buffer_size();
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("temporary buffer")));
  package.mutable_entrypoint(0)->set_temp_buffer_size(-1);
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("temporary buffer")));
  package.mutable_entrypoint(0)->set_temp_buffer_size(64);
  package.mutable_entrypoint(0)->set_temp_buffer_alignment(3);
  EXPECT_THAT(Generate(package),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("zero or a power of two")));
  package.mutable_entrypoint(0)->set_temp_buffer_alignment(0);
  EXPECT_TRUE(Generate(package).ok());
}

TEST_F(AotCppHeaderTest, MissingFunctionMetadataErrors) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(8)}}, Bits(8));
  package.mutable_entrypoint(0)->clear_function_metadata();
  EXPECT_THAT(Generate(package),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("no function metadata")));
}

TEST_F(AotCppHeaderTest, MalformedTypeDimensionsError) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(8)}}, Bits(8));
  package.mutable_entrypoint(0)
      ->mutable_function_metadata()
      ->mutable_function_interface()
      ->mutable_parameters(0)
      ->mutable_type()
      ->set_bit_count(-1);
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("negative width")));
}

TEST_F(AotCppHeaderTest, PackedSizeMismatchErrors) {
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"x", Bits(3)}}, Bits(8));
  package.mutable_entrypoint(0)->set_packed_input_buffer_sizes(0, 99);
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("Packed size mismatch")));
}

TEST_F(AotCppHeaderTest, UnsupportedTypeKindsError) {
  TypeProto token;
  token.set_type_enum(TypeProto::TOKEN);
  AotPackageEntrypointsProto package =
      MakeFunctionPackage({{"tok", token}}, Bits(8));
  EXPECT_THAT(Generate(package), StatusIs(absl::StatusCode::kUnimplemented,
                                          HasSubstr("only supports BITS")));
}

}  // namespace
}  // namespace xls
