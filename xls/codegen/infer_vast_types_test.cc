// Copyright 2024 The XLS Authors
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

#include "xls/codegen/infer_vast_types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {
namespace {

class InferVastTypesTest : public ::testing::Test {
 public:
  InferVastTypesTest() : file_(FileType::kSystemVerilog) {
    module_ = file_.AddModule("test_module", SourceInfo());
    uint16_max_ = file_.Make<Literal>(
        SourceInfo(), Bits::AllOnes(16), FormatPreference::kDefault,
        /*declared_bit_count=*/16, /*emit_bit_count=*/true,
        /*declared_as_signed=*/false);
  }

  std::string InferTypesToString() {
    absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> map =
        InferVastTypes(&file_);
    XLS_EXPECT_OK(map);
    return TypesToString(*map);
  }

  std::string InferTypesToString(Expression* expr) {
    absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> map =
        InferVastTypes(expr);
    XLS_EXPECT_OK(map);
    return TypesToString(*map);
  }

  std::string TypesToString(
      const absl::flat_hash_map<Expression*, DataType*>& map) {
    // The aim here is to convert the map into strings in the form "expr : type"
    // in an order that is roughly topological, without having a real topo sort.
    std::vector<std::string> strings;
    strings.reserve(map.size());
    for (const auto& [expr, data_type] : map) {
      // A type to string conversion is not always exactly like emitting
      // Verilog, because the logic for the latter assumes it's paired with a
      // DataKind and identifier.
      std::string type_string;
      if (dynamic_cast<UnpackedArrayType*>(data_type)) {
        type_string = data_type->EmitWithIdentifier(nullptr, "");
      } else {
        type_string = data_type->Emit(nullptr);
      }
      if (type_string.empty()) {
        if (dynamic_cast<IntegerType*>(data_type) != nullptr) {
          type_string = "integer";
        } else if (dynamic_cast<ScalarType*>(data_type) != nullptr) {
          type_string = "logic";
        }
      }
      strings.push_back(
          absl::StrFormat("%s : %s", expr->Emit(nullptr),
                          absl::StripAsciiWhitespace(type_string)));
    }
    std::sort(strings.begin(), strings.end(),
              [](const std::string& a, const std::string& b) {
                size_t a_expr_length = a.find(':');
                size_t b_expr_length = b.find(':');
                int64_t literal_test = 0;
                const bool a_literal = absl::SimpleAtoi(
                    absl::StripAsciiWhitespace(a.substr(0, a_expr_length)),
                    &literal_test);
                const bool b_literal = absl::SimpleAtoi(
                    absl::StripAsciiWhitespace(b.substr(0, b_expr_length)),
                    &literal_test);
                if (a_literal ^ b_literal) {
                  return a_literal;
                }
                if (a_expr_length != b_expr_length) {
                  return a_expr_length < b_expr_length;
                }
                return a < b;
              });
    std::string result;
    for (const std::string& next : strings) {
      if (strings.size() > 1) {
        absl::StrAppend(&result, "\n");
      }
      absl::StrAppend(&result, next);
    }
    if (strings.size() > 1) {
      absl::StrAppend(&result, "\n");
    }
    return result;
  }

  Literal* UnsignedZero(int bits) {
    return file_.Make<Literal>(
        SourceInfo(), Bits(bits), FormatPreference::kDefault,
        /*declared_bit_count=*/bits, /*emit_bit_count=*/true,
        /*declared_as_signed=*/false);
  }

  Literal* BareLiteral(int32_t value) {
    absl::StatusOr<Bits> bits = ParseNumber(std::to_string(value));
    CHECK_OK(bits);
    return file_.Make<Literal>(SourceInfo(),
                               value < 0 ? bits_ops::SignExtend(*bits, 32)
                                         : bits_ops::ZeroExtend(*bits, 32),
                               FormatPreference::kDefault,
                               /*declared_bit_count=*/32,
                               /*emit_bit_count=*/true,
                               /*declared_as_signed=*/true);
  }

  Literal* Uint64Literal(uint64_t value) {
    absl::StatusOr<Bits> bits = ParseNumber(std::to_string(value));
    CHECK_OK(bits);
    return file_.Make<Literal>(SourceInfo(), bits_ops::ZeroExtend(*bits, 64),
                               FormatPreference::kHex,
                               /*declared_bit_count=*/64,
                               /*emit_bit_count=*/true,
                               /*declared_as_signed=*/false);
  }

  VerilogFile file_;
  Module* module_;
  Literal* uint16_max_;
  Literal* u3_zero_;
  Literal* u4_zero_;
};

TEST_F(InferVastTypesTest, PlainLiteralParameter) {
  SourceInfo loc;
  module_->AddParameter("foo", file_.PlainLiteral(3, loc), loc);
  file_.PlainLiteral(3, SourceInfo());
  EXPECT_EQ(InferTypesToString(), "3 : integer");
}

TEST_F(InferVastTypesTest, BrokenShiftAddExampleFromSpec) {
  // answer = (a + b) >> 1, where all parameters are logic [15:0].
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(16, loc)),
      uint16_max_, loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(16, loc)),
      uint16_max_, loc);
  module_->AddParameter(
      file_.Make<Def>(loc, "answer", DataKind::kLogic,
                      file_.BitVectorType(16, loc)),
      file_.Shrl(file_.Add(a, b, loc), file_.PlainLiteral(1, loc), loc), loc);
  EXPECT_EQ(InferTypesToString(), R"(
1 : integer
65535 : [15:0]
a : [15:0]
b : [15:0]
a + b : [15:0]
a + b >> 1 : [15:0]
)");
}

TEST_F(InferVastTypesTest, FixedShiftAddExampleFromSpec) {
  // answer = (a + b + 0) >> 1, where all parameters are logic [15:0].
  // The zero, being s32 as a bare literal, should promote the size of a and b.
  // However, the unsigned a and b undo the automatic signedness of the 0, by
  // the rules in 11.8.1.
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(16, loc)),
      uint16_max_, loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(16, loc)),
      uint16_max_, loc);
  module_->AddParameter(
      file_.Make<Def>(loc, "answer", DataKind::kLogic,
                      file_.BitVectorType(16, loc)),
      file_.Shrl(file_.Add(a, file_.Add(b, BareLiteral(0), loc), loc),
                 file_.PlainLiteral(1, loc), loc),
      loc);
  EXPECT_EQ(InferTypesToString(), R"(
0 : unsigned
1 : integer
65535 : [15:0]
a : unsigned
b : unsigned
b + 0 : unsigned
a + (b + 0) : unsigned
a + (b + 0) >> 1 : [15:0]
)");
}

TEST_F(InferVastTypesTest, TernaryExampleFromSpec) {
  // c ? (a & b) : d, where all are 3 bits, except d, which is 4 bits.
  // d should promote the whole other branch of the ternary, but not the
  // condition.
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  auto* c = module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  auto* d = module_->AddParameter(
      file_.Make<Def>(loc, "d", DataKind::kLogic, file_.BitVectorType(5, loc)),
      UnsignedZero(5), loc);
  EXPECT_EQ(
      InferTypesToString(file_.Ternary(c, file_.BitwiseAnd(a, b, loc), d, loc)),
      R"(
a : [4:0]
b : [4:0]
c : [3:0]
d : [4:0]
a & b : [4:0]
c ? a & b : d : [4:0]
)");
}

TEST_F(InferVastTypesTest, ComplexTernary) {
  // (a > b) ? a : ((b < c) ? b : c)
  // where a is 3 bits, b is 4 bits, and c is 5 bits. The point here is mainly
  // to make sure all the refs to each variable get their own inferred type.
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(3, loc)),
      UnsignedZero(3), loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  auto* c = module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic, file_.BitVectorType(5, loc)),
      UnsignedZero(5), loc);
  EXPECT_EQ(InferTypesToString(file_.Ternary(
                file_.GreaterThan(a, b, loc), a->Duplicate(),
                file_.Ternary(file_.LessThan(b->Duplicate(), c, loc),
                              b->Duplicate(), c->Duplicate(), loc),
                loc)),
            R"(
a : [3:0]
a : [4:0]
b : [3:0]
b : [4:0]
b : [4:0]
c : [4:0]
c : [4:0]
a > b : logic
b < c : logic
a > b ? a : (b < c ? b : c) : [4:0]
b < c ? b : c : [4:0]
)");
}

TEST_F(InferVastTypesTest, SimpleMultiplicationFromSpec) {
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(6, loc)),
      UnsignedZero(6), loc);
  EXPECT_EQ(InferTypesToString(file_.Mul(a, b, loc)),
            R"(
a : [5:0]
b : [5:0]
a * b : [5:0]
)");
}

TEST_F(InferVastTypesTest, PowerWrappedInConcatFromSpec) {
  // c = {a ** b}, with a..c being of increasing width. The concat blocks
  // promotion, and the power operator doesn't promote its RHS, so nothing is
  // affected.
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(6, loc)),
      UnsignedZero(6), loc);
  module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic, file_.BitVectorType(16, loc)),
      file_.Concat({file_.Power(a, b, loc)}, loc), loc);
  EXPECT_EQ(InferTypesToString(), R"(
0 : [3:0]
0 : [5:0]
a : [3:0]
b : [5:0]
a ** b : [3:0]
{a ** b} : [15:0]
)");
}

TEST_F(InferVastTypesTest, PowerNotWrappedInConcatFromSpec) {
  // c = a ** b, with a..c being of increasing width. c should promote a but
  // not b.
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(6, loc)),
      UnsignedZero(6), loc);
  module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic, file_.BitVectorType(16, loc)),
      file_.Power(a, b, loc), loc);
  EXPECT_EQ(InferTypesToString(), R"(
0 : [3:0]
0 : [5:0]
a : [15:0]
b : [5:0]
a ** b : [15:0]
)");
}

TEST_F(InferVastTypesTest, MultiFile) {
  SourceInfo loc;
  VerilogFile file2(FileType::kSystemVerilog);
  Module* module2 = file2.AddModule("module2", loc);
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(4, loc)),
      UnsignedZero(4), loc);
  module2->AddParameter(
      file2.Make<Def>(loc, "b", DataKind::kLogic, file2.BitVectorType(6, loc)),
      file2.Add(file2.PlainLiteral(50, loc), a, loc), loc);
  std::vector<VerilogFile*> files = {&file_, &file2};
  auto types = InferVastTypes(files);
  XLS_ASSERT_OK(types);
  EXPECT_EQ(TypesToString(*types),
            R"(
0 : [3:0]
50 : unsigned
a : unsigned
50 + a : [5:0]
)");
}

TEST_F(InferVastTypesTest, BigConstants) {
  SourceInfo loc;
  // parameter int unsigned GiB = 1024 * 1024 * 1024;
  // Note that according to 11.8.1, it appears the automatic signedness of the
  // 1024's is not affected by the unsigned LHS until we are ready to actually
  // put the result into the LHS.
  auto* gib = module_->AddParameter(
      file_.Make<Def>(loc, "GiB", DataKind::kInteger,
                      file_.Make<IntegerType>(loc, /*signed=*/false)),
      file_.Mul(BareLiteral(1024),
                file_.Mul(BareLiteral(1024), BareLiteral(1024), loc), loc),
      loc);
  // parameter logic [63:0] hundredGiB = 100 * GiB;
  module_->AddParameter(file_.Make<Def>(loc, "hundredGiB", DataKind::kLogic,
                                        file_.BitVectorType(64, loc)),
                        file_.Mul(BareLiteral(100), gib, loc), loc);
  // parameter logic [63:0] other_hundredGiB = 64'h1900000000;
  module_->AddParameter(
      file_.Make<Def>(loc, "other_hundredGiB", DataKind::kLogic,
                      file_.BitVectorType(64, loc)),
      Uint64Literal(0x1900000000), loc);
  // Note: the GiB that appears here is the rvalue in the hundredGiB line.
  EXPECT_EQ(InferTypesToString(), R"(
100 : [63:0]
1024 : integer
1024 : integer
1024 : integer
GiB : [63:0]
100 * GiB : [63:0]
1024 * 1024 : integer
1024 * (1024 * 1024) : unsigned
64'h0000_0019_0000_0000 : [63:0]
)");
}

TEST_F(InferVastTypesTest, NonLiteralArrayDim) {
  SourceInfo loc;
  // parameter foo = 24;
  // parameter logic[foo - 1:0] bar = 55;
  // parameter logic[63:0] baz = bar + 1;
  auto* foo = module_->AddParameter("foo", BareLiteral(24), loc);
  auto* bar = module_->AddParameter(
      file_.Make<Def>(loc, "bar", DataKind::kLogic,
                      file_.Make<BitVectorType>(
                          loc, file_.Sub(foo, BareLiteral(1), loc),
                          /*is_signed=*/false, /*size_expr_is_max=*/true)),
      BareLiteral(55), loc);
  module_->AddParameter(
      file_.Make<Def>(
          loc, "baz", DataKind::kLogic,
          file_.Make<BitVectorType>(loc, BareLiteral(63), /*is_signed=*/false,
                                    /*size_expr_is_max=*/true)),
      file_.Add(bar, BareLiteral(1), loc), loc);
  EXPECT_EQ(InferTypesToString(), R"(
1 : [63:0]
1 : integer
24 : integer
55 : [23:0]
bar : [63:0]
foo : integer
bar + 1 : [63:0]
foo - 1 : integer
)");
}

TEST_F(InferVastTypesTest, FunctionCall) {
  // function automatic logic[15:0] fn(
  //    logic[7:0] a,
  //    logic[63:0] b);
  //   ...
  // endfunction
  // 1 + fn(50, 25 + 4'0)
  SourceInfo loc;
  VerilogFunction* fn =
      file_.Make<VerilogFunction>(loc, "fn", file_.BitVectorType(16, loc));
  fn->AddArgument(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(8, loc)),
      loc);
  fn->AddArgument(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(64, loc)),
      loc);
  EXPECT_EQ(InferTypesToString(file_.Add(
                BareLiteral(1),
                file_.Make<VerilogFunctionCall>(
                    loc, fn,
                    std::vector<Expression*>{
                        BareLiteral(50),
                        file_.Add(BareLiteral(25), UnsignedZero(4), loc)}),
                loc)),
            R"(
0 : [63:0]
1 : unsigned
25 : [63:0]
50 : [7:0]
25 + 0 : [63:0]
fn(50, 25 + 0) : unsigned
1 + fn(50, 25 + 0) : unsigned
)");
}

TEST_F(InferVastTypesTest, SystemFunctionCall) {
  // 1 + clog2(25 + 4'0)
  SourceInfo loc;
  EXPECT_EQ(InferTypesToString(
                file_.Add(BareLiteral(1),
                          file_.Make<SystemFunctionCall>(
                              loc, "clog2",
                              std::vector<Expression*>{file_.Add(
                                  BareLiteral(25), UnsignedZero(4), loc)}),
                          loc)),
            R"(
0 : unsigned
1 : integer
25 : unsigned
25 + 0 : unsigned
$clog2(25 + 0) : integer
1 + $clog2(25 + 0) : integer
)");
}

TEST_F(InferVastTypesTest, EnumPromotion) {
  // typedef enum logic[11:0] {
  //   foo = 2
  // } enum_t;
  // parameter unsigned foo_plus_one = foo + 1;
  SourceInfo loc;
  Enum* enum_def =
      file_.Make<Enum>(loc, DataKind::kLogic, file_.BitVectorType(12, loc));
  EnumMemberRef* foo = enum_def->AddMember("foo", BareLiteral(2), loc);
  Typedef* type_def = file_.Make<Typedef>(
      loc, file_.Make<Def>(loc, "enum_t", DataKind::kUser, enum_def));
  module_->AddModuleMember(type_def);
  module_->AddParameter(
      file_.Make<Def>(loc, "foo_plus_one", DataKind::kLogic,
                      file_.Make<IntegerType>(loc, /*signed=*/false)),
      file_.Add(foo, BareLiteral(1), loc), loc);
  EXPECT_EQ(InferTypesToString(),
            R"(
1 : unsigned
2 : [11:0]
foo : unsigned
foo + 1 : unsigned
)");
}

TEST_F(InferVastTypesTest, PromotionToFoldedTypedef) {
  // parameter size = 50;
  // typedef logic[size - 1:0] foo_t;
  // parameter mb = 1024 * 1024;
  // parameter foo_t bar = 50 * mb;
  SourceInfo loc;
  auto* size = module_->AddParameter("size", BareLiteral(50), loc);
  Typedef* type_def = file_.Make<Typedef>(
      loc,
      file_.Make<Def>(loc, "foo_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(
                          loc, file_.Sub(size, BareLiteral(1), loc),
                          /*is_signed=*/false, /*size_expr_is_max=*/true)));
  module_->AddModuleMember(type_def);
  auto* mb = module_->AddParameter(
      "mb", file_.Mul(BareLiteral(1024), BareLiteral(1024), loc), loc);
  module_->AddParameter(file_.Make<Def>(loc, "bar", DataKind::kUser,
                                        file_.Make<TypedefType>(loc, type_def)),
                        file_.Mul(BareLiteral(50), mb, loc), loc);
  EXPECT_EQ(InferTypesToString(),
            R"(
1 : integer
50 : [49:0]
50 : integer
1024 : integer
1024 : integer
mb : [49:0]
size : integer
50 * mb : [49:0]
size - 1 : integer
1024 * 1024 : integer
)");
}

TEST_F(InferVastTypesTest, ConcatWithFoldedTypedef) {
  // parameter size = 50;
  // typedef logic[size - 1:0] foo_t;
  // parameter foo_t foo = 3;
  // parameter integer bar = 4;
  // {foo, bar}
  SourceInfo loc;
  auto* size = module_->AddParameter("size", BareLiteral(50), loc);
  Typedef* type_def = file_.Make<Typedef>(
      loc,
      file_.Make<Def>(loc, "foo_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(
                          loc, file_.Sub(size, BareLiteral(1), loc),
                          /*is_signed=*/false, /*size_expr_is_max=*/true)));
  module_->AddModuleMember(type_def);
  auto* foo = module_->AddParameter(
      file_.Make<Def>(loc, "foo", DataKind::kUser,
                      file_.Make<TypedefType>(loc, type_def)),
      BareLiteral(3), loc);
  auto* bar =
      module_->AddParameter(file_.Make<Def>(loc, "bar", DataKind::kInteger,
                                            file_.Make<IntegerType>(loc)),
                            BareLiteral(4), loc);
  EXPECT_EQ(
      InferTypesToString(file_.Concat(std::vector<Expression*>{foo, bar}, loc)),
      R"(
bar : integer
foo : foo_t
{foo, bar} : [81:0]
)");
}

TEST_F(InferVastTypesTest, PackedStruct) {
  // typedef struct packed {
  //   logic foo;
  //   integer bar;
  //   logic [15:0] baz;
  // } struct_t;
  // parameter struct_t a;
  // parameter logic[63:0] b;
  // parameter logic[23:0] c;
  // a + b should promote a
  // a + c should promote c
  SourceInfo loc;
  Struct* struct_def = file_.Make<Struct>(
      loc, std::vector<Def*>{file_.Make<Def>(loc, "foo", DataKind::kLogic,
                                             file_.ScalarType(loc)),
                             file_.Make<Def>(loc, "bar", DataKind::kInteger,
                                             file_.IntegerType(loc)),
                             file_.Make<Def>(loc, "baz", DataKind::kLogic,
                                             file_.BitVectorType(16, loc))});
  Typedef* type_def = file_.Make<Typedef>(
      loc, file_.Make<Def>(loc, "struct_t", DataKind::kUser, struct_def));
  module_->AddModuleMember(type_def);
  ParameterRef* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kUser,
                      file_.Make<TypedefType>(loc, type_def)),
      BareLiteral(0), loc);
  ParameterRef* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(64, loc)),
      BareLiteral(0), loc);
  ParameterRef* c = module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic, file_.BitVectorType(23, loc)),
      BareLiteral(0), loc);
  EXPECT_EQ(InferTypesToString(file_.Add(a, b, loc)),
            R"(
a : [63:0]
b : [63:0]
a + b : [63:0]
)");
  EXPECT_EQ(InferTypesToString(file_.Add(a, c, loc)),
            R"(
a : [48:0]
c : [48:0]
a + c : [48:0]
)");
}

TEST_F(InferVastTypesTest, UnpackedArray) {
  // reg[15:0] a[2][4];
  // reg[15:0] b[2][4];
  // parameter logic [15:0] c[2][4] = a;
  // a == b
  // There's not much you can do with unpacked arrays that is valid. We
  // basically want to make sure we don't crash on them.
  SourceInfo loc;
  LogicRef* a =
      module_->AddReg("a", file_.UnpackedArrayType(16, {2, 4}, loc), loc);
  LogicRef* b =
      module_->AddReg("b", file_.UnpackedArrayType(16, {2, 4}, loc), loc);
  module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic,
                      file_.UnpackedArrayType(16, {2, 4}, loc)),
      a, loc);

  // The `a` ref in the init of `c`.
  EXPECT_EQ(InferTypesToString(), "a : [15:0] [2][4]");

  EXPECT_EQ(InferTypesToString(file_.Equals(a, b, loc)),
            R"(
a : [15:0] [2][4]
b : [15:0] [2][4]
a == b : logic
)");
}

TEST_F(InferVastTypesTest, ReturnTypeCoercion) {
  // function automatic logic[15:0] fn(
  //    logic[23:0] a);
  //   return a + 1;
  // endfunction
  SourceInfo loc;
  VerilogFunction* fn =
      file_.Make<VerilogFunction>(loc, "fn", file_.BitVectorType(16, loc));
  LogicRef* a = fn->AddArgument(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(24, loc)),
      loc);
  fn->AddStatement<ReturnStatement>(loc, file_.Add(a, BareLiteral(1), loc));
  module_->top()->AddModuleMember(fn);
  EXPECT_EQ(InferTypesToString(),
            R"(
1 : unsigned
a : unsigned
a + 1 : [15:0]
)");
}

TEST_F(InferVastTypesTest, TypedefReturnType) {
  // parameter width = 24;
  // typedef logic[width - 1:0] word_t;
  // function automatic word_t fn(
  //    word_t a);
  //   return a + 3'0;
  // endfunction
  // Getting this right requires constant folding the definition of `word_t` and
  // promoting the 3-bit value.
  SourceInfo loc;
  auto* width = module_->AddParameter("width", BareLiteral(24), loc);
  Typedef* word_t = module_->AddTypedef(
      file_.Make<Def>(loc, "word_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(
                          loc, file_.Sub(width, BareLiteral(1), loc),
                          /*is_signed=*/false, /*size_expr_is_max=*/true)),
      loc);
  TypedefType* word_t_type = file_.Make<TypedefType>(loc, word_t);
  VerilogFunction* fn = file_.Make<VerilogFunction>(loc, "fn", word_t_type);
  LogicRef* a = fn->AddArgument(
      file_.Make<Def>(loc, "a", DataKind::kUser, word_t_type), loc);
  fn->AddStatement<ReturnStatement>(loc, file_.Add(a, UnsignedZero(3), loc));
  module_->top()->AddModuleMember(fn);
  EXPECT_EQ(InferTypesToString(),
            R"(
0 : [23:0]
1 : integer
24 : integer
a : [23:0]
a + 0 : [23:0]
width : integer
width - 1 : integer
)");
}

TEST_F(InferVastTypesTest, ContextDependentUnary) {
  // b = ~(a + 0) where all parameters are logic [15:0].
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(16, loc)),
      uint16_max_, loc);
  module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(16, loc)),
      file_.BitwiseNot(file_.Add(a, BareLiteral(0), loc), loc), loc);
  EXPECT_EQ(InferTypesToString(), R"(
0 : unsigned
65535 : [15:0]
a : unsigned
a + 0 : unsigned
~(a + 0) : [15:0]
)");
}

TEST_F(InferVastTypesTest, Comparison) {
  // a + (b > c)
  // where a is 1 bit, b is 3 bits, and c is 2 bits. c should be promoted but
  // the comparison itself should have a scalar type.
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.ScalarType(loc)),
      BareLiteral(0), loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(3, loc)),
      BareLiteral(0), loc);
  auto* c = module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic, file_.BitVectorType(2, loc)),
      BareLiteral(0), loc);
  EXPECT_EQ(InferTypesToString(file_.Add(a, file_.GreaterThan(b, c, loc), loc)),
            R"(
a : logic
b : [2:0]
c : [2:0]
b > c : logic
a + (b > c) : logic
)");
}

TEST_F(InferVastTypesTest, LogicalAnd) {
  // a && (b + 1)
  // where a and b are 16 bits. b should be promoted and a should not.
  SourceInfo loc;
  auto* a = module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(16, loc)),
      BareLiteral(0), loc);
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(16, loc)),
      BareLiteral(0), loc);
  EXPECT_EQ(InferTypesToString(
                file_.LogicalAnd(a, file_.Add(b, BareLiteral(1), loc), loc)),
            R"(
1 : unsigned
a : [15:0]
b : unsigned
b + 1 : unsigned
a && b + 1 : logic
)");
}

TEST_F(InferVastTypesTest, SelfDeterminedUnary) {
  // a = &(b + c), where a and b are 16 bits, and c is 32 bits. b should be
  // promoted, but b + c is an independent expr from the rest.
  SourceInfo loc;
  auto* b = module_->AddParameter(
      file_.Make<Def>(loc, "b", DataKind::kLogic, file_.BitVectorType(16, loc)),
      BareLiteral(0), loc);
  auto* c = module_->AddParameter(
      file_.Make<Def>(loc, "c", DataKind::kLogic, file_.BitVectorType(32, loc)),
      BareLiteral(0), loc);
  module_->AddParameter(
      file_.Make<Def>(loc, "a", DataKind::kLogic, file_.BitVectorType(16, loc)),
      file_.AndReduce(file_.Add(b, c, loc), loc), loc);
  EXPECT_EQ(InferTypesToString(),
            R"(
0 : [15:0]
0 : [31:0]
b : [31:0]
c : [31:0]
b + c : [31:0]
&(b + c) : [15:0]
)");
}

TEST_F(InferVastTypesTest, UseOfUntypedParameter) {
  SourceInfo loc;
  auto* foo = module_->AddParameter("foo", UnsignedZero(4), loc);
  module_->AddParameter("bar", file_.Add(UnsignedZero(2), foo, loc), loc);
  file_.PlainLiteral(3, SourceInfo());
  EXPECT_EQ(InferTypesToString(), R"(
0 : [3:0]
0 : [3:0]
foo : [3:0]
0 + foo : [3:0]
)");
}

}  // namespace
}  // namespace verilog
}  // namespace xls
