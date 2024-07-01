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

#include "xls/codegen/vast/infer_vast_types.h"

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
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/fileno.h"
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

  SourceInfo NextLoc() {
    return SourceInfo(
        SourceLocation(Fileno(0), Lineno(++next_loc_lineno_), Colno(0)));
  }

  VerilogFile file_;
  int next_loc_lineno_ = 0;
  Module* module_;
  Literal* uint16_max_;
  Literal* u3_zero_;
  Literal* u4_zero_;
};

TEST_F(InferVastTypesTest, PlainLiteralParameter) {
  module_->AddParameter("foo", file_.PlainLiteral(3, NextLoc()), NextLoc());
  file_.PlainLiteral(3, SourceInfo());
  EXPECT_EQ(InferTypesToString(), "3 : integer");
}

TEST_F(InferVastTypesTest, BrokenShiftAddExampleFromSpec) {
  // answer = (a + b) >> 1, where all parameters are logic [15:0].
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            uint16_max_, NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            uint16_max_, NextLoc());
  module_->AddParameter(file_.Make<Def>(NextLoc(), "answer", DataKind::kLogic,
                                        file_.BitVectorType(16, NextLoc())),
                        file_.Shrl(file_.Add(a, b, NextLoc()),
                                   file_.PlainLiteral(1, NextLoc()), NextLoc()),
                        NextLoc());
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            uint16_max_, NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            uint16_max_, NextLoc());
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "answer", DataKind::kLogic,
                      file_.BitVectorType(16, NextLoc())),
      file_.Shrl(
          file_.Add(a, file_.Add(b, BareLiteral(0), NextLoc()), NextLoc()),
          file_.PlainLiteral(1, NextLoc()), NextLoc()),
      NextLoc());
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  auto* c =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  auto* d =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "d", DataKind::kLogic,
                                            file_.BitVectorType(5, NextLoc())),
                            UnsignedZero(5), NextLoc());
  EXPECT_EQ(InferTypesToString(file_.Ternary(
                c, file_.BitwiseAnd(a, b, NextLoc()), d, NextLoc())),
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(3, NextLoc())),
                            UnsignedZero(3), NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  auto* c =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                                            file_.BitVectorType(5, NextLoc())),
                            UnsignedZero(5), NextLoc());
  EXPECT_EQ(InferTypesToString(file_.Ternary(
                file_.GreaterThan(a, b, NextLoc()), a->Duplicate(),
                file_.Ternary(file_.LessThan(b->Duplicate(), c, NextLoc()),
                              b->Duplicate(), c->Duplicate(), NextLoc()),
                NextLoc())),
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(6, NextLoc())),
                            UnsignedZero(6), NextLoc());
  EXPECT_EQ(InferTypesToString(file_.Mul(a, b, NextLoc())),
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(6, NextLoc())),
                            UnsignedZero(6), NextLoc());
  module_->AddParameter(file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                                        file_.BitVectorType(16, NextLoc())),
                        file_.Concat({file_.Power(a, b, NextLoc())}, NextLoc()),
                        NextLoc());
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(6, NextLoc())),
                            UnsignedZero(6), NextLoc());
  module_->AddParameter(file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                                        file_.BitVectorType(16, NextLoc())),
                        file_.Power(a, b, NextLoc()), NextLoc());
  EXPECT_EQ(InferTypesToString(), R"(
0 : [3:0]
0 : [5:0]
a : [15:0]
b : [5:0]
a ** b : [15:0]
)");
}

TEST_F(InferVastTypesTest, MultiFile) {
  VerilogFile file2(FileType::kSystemVerilog);
  Module* module2 = file2.AddModule("module2", NextLoc());
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(4, NextLoc())),
                            UnsignedZero(4), NextLoc());
  module2->AddParameter(
      file2.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                      file2.BitVectorType(6, NextLoc())),
      file2.Add(file2.PlainLiteral(50, NextLoc()), a, NextLoc()), NextLoc());
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
  // parameter int unsigned GiB = 1024 * 1024 * 1024;
  // Note that according to 11.8.1, it appears the automatic signedness of the
  // 1024's is not affected by the unsigned LHS until we are ready to actually
  // put the result into the LHS.
  auto* gib = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "GiB", DataKind::kInteger,
                      file_.Make<IntegerType>(NextLoc(), /*signed=*/false)),
      file_.Mul(BareLiteral(1024),
                file_.Mul(BareLiteral(1024), BareLiteral(1024), NextLoc()),
                NextLoc()),
      NextLoc());
  // parameter logic [63:0] hundredGiB = 100 * GiB;
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "hundredGiB", DataKind::kLogic,
                      file_.BitVectorType(64, NextLoc())),
      file_.Mul(BareLiteral(100), gib, NextLoc()), NextLoc());
  // parameter logic [63:0] other_hundredGiB = 64'h1900000000;
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "other_hundredGiB", DataKind::kLogic,
                      file_.BitVectorType(64, NextLoc())),
      Uint64Literal(0x1900000000), NextLoc());
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
  // parameter foo = 24;
  // parameter logic[foo - 1:0] bar = 55;
  // parameter logic[63:0] baz = bar + 1;
  auto* foo = module_->AddParameter("foo", BareLiteral(24), NextLoc());
  auto* bar = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "bar", DataKind::kLogic,
                      file_.Make<BitVectorType>(
                          NextLoc(), file_.Sub(foo, BareLiteral(1), NextLoc()),
                          /*is_signed=*/false, /*size_expr_is_max=*/true)),
      BareLiteral(55), NextLoc());
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "baz", DataKind::kLogic,
                      file_.Make<BitVectorType>(NextLoc(), BareLiteral(63),
                                                /*is_signed=*/false,
                                                /*size_expr_is_max=*/true)),
      file_.Add(bar, BareLiteral(1), NextLoc()), NextLoc());
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
  VerilogFunction* fn = file_.Make<VerilogFunction>(
      NextLoc(), "fn", file_.BitVectorType(16, NextLoc()));
  fn->AddArgument(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                  file_.BitVectorType(8, NextLoc())),
                  NextLoc());
  fn->AddArgument(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                  file_.BitVectorType(64, NextLoc())),
                  NextLoc());
  EXPECT_EQ(
      InferTypesToString(file_.Add(
          BareLiteral(1),
          file_.Make<VerilogFunctionCall>(
              NextLoc(), fn,
              std::vector<Expression*>{
                  BareLiteral(50),
                  file_.Add(BareLiteral(25), UnsignedZero(4), NextLoc())}),
          NextLoc())),
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
  EXPECT_EQ(InferTypesToString(file_.Add(
                BareLiteral(1),
                file_.Make<SystemFunctionCall>(
                    NextLoc(), "clog2",
                    std::vector<Expression*>{file_.Add(
                        BareLiteral(25), UnsignedZero(4), NextLoc())}),
                NextLoc())),
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
  Enum* enum_def = file_.Make<Enum>(NextLoc(), DataKind::kLogic,
                                    file_.BitVectorType(12, NextLoc()));
  EnumMemberRef* foo = enum_def->AddMember("foo", BareLiteral(2), NextLoc());
  Typedef* type_def = file_.Make<Typedef>(
      NextLoc(),
      file_.Make<Def>(NextLoc(), "enum_t", DataKind::kUser, enum_def));
  module_->AddModuleMember(type_def);
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "foo_plus_one", DataKind::kLogic,
                      file_.Make<IntegerType>(NextLoc(), /*signed=*/false)),
      file_.Add(foo, BareLiteral(1), NextLoc()), NextLoc());
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
  auto* size = module_->AddParameter("size", BareLiteral(50), NextLoc());
  Typedef* type_def = file_.Make<Typedef>(
      NextLoc(),
      file_.Make<Def>(NextLoc(), "foo_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(
                          NextLoc(), file_.Sub(size, BareLiteral(1), NextLoc()),
                          /*is_signed=*/false, /*size_expr_is_max=*/true)));
  module_->AddModuleMember(type_def);
  auto* mb = module_->AddParameter(
      "mb", file_.Mul(BareLiteral(1024), BareLiteral(1024), NextLoc()),
      NextLoc());
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "bar", DataKind::kUser,
                      file_.Make<TypedefType>(NextLoc(), type_def)),
      file_.Mul(BareLiteral(50), mb, NextLoc()), NextLoc());
  EXPECT_EQ(InferTypesToString(),
            R"(
1 : integer
50 : foo_t
50 : integer
1024 : integer
1024 : integer
mb : foo_t
size : integer
50 * mb : foo_t
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
  auto* size = module_->AddParameter("size", BareLiteral(50), NextLoc());
  Typedef* type_def = file_.Make<Typedef>(
      NextLoc(),
      file_.Make<Def>(NextLoc(), "foo_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(
                          NextLoc(), file_.Sub(size, BareLiteral(1), NextLoc()),
                          /*is_signed=*/false, /*size_expr_is_max=*/true)));
  module_->AddModuleMember(type_def);
  auto* foo = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "foo", DataKind::kUser,
                      file_.Make<TypedefType>(NextLoc(), type_def)),
      BareLiteral(3), NextLoc());
  auto* bar = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "bar", DataKind::kInteger,
                      file_.Make<IntegerType>(NextLoc())),
      BareLiteral(4), NextLoc());
  EXPECT_EQ(InferTypesToString(
                file_.Concat(std::vector<Expression*>{foo, bar}, NextLoc())),
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
  Struct* struct_def = file_.Make<Struct>(
      NextLoc(),
      std::vector<Def*>{file_.Make<Def>(NextLoc(), "foo", DataKind::kLogic,
                                        file_.ScalarType(NextLoc())),
                        file_.Make<Def>(NextLoc(), "bar", DataKind::kInteger,
                                        file_.IntegerType(NextLoc())),
                        file_.Make<Def>(NextLoc(), "baz", DataKind::kLogic,
                                        file_.BitVectorType(16, NextLoc()))});
  Typedef* type_def = file_.Make<Typedef>(
      NextLoc(),
      file_.Make<Def>(NextLoc(), "struct_t", DataKind::kUser, struct_def));
  module_->AddModuleMember(type_def);
  ParameterRef* a = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "a", DataKind::kUser,
                      file_.Make<TypedefType>(NextLoc(), type_def)),
      BareLiteral(0), NextLoc());
  ParameterRef* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(64, NextLoc())),
                            BareLiteral(0), NextLoc());
  ParameterRef* c =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                                            file_.BitVectorType(23, NextLoc())),
                            BareLiteral(0), NextLoc());
  EXPECT_EQ(InferTypesToString(file_.Add(a, b, NextLoc())),
            R"(
a : [63:0]
b : [63:0]
a + b : [63:0]
)");
  EXPECT_EQ(InferTypesToString(file_.Add(a, c, NextLoc())),
            R"(
a : struct_t
c : struct_t
a + c : struct_t
)");
}

TEST_F(InferVastTypesTest, UnpackedArray) {
  // reg[15:0] a[2][4];
  // reg[15:0] b[2][4];
  // parameter logic [15:0] c[2][4] = a;
  // a == b
  // There's not much you can do with unpacked arrays that is valid. We
  // basically want to make sure we don't crash on them.
  LogicRef* a = module_->AddReg(
      "a", file_.UnpackedArrayType(16, {2, 4}, NextLoc()), NextLoc());
  LogicRef* b = module_->AddReg(
      "b", file_.UnpackedArrayType(16, {2, 4}, NextLoc()), NextLoc());
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                      file_.UnpackedArrayType(16, {2, 4}, NextLoc())),
      a, NextLoc());

  // The `a` ref in the init of `c`.
  EXPECT_EQ(InferTypesToString(), "a : [15:0] [2][4]");

  EXPECT_EQ(InferTypesToString(file_.Equals(a, b, NextLoc())),
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
  VerilogFunction* fn = file_.Make<VerilogFunction>(
      NextLoc(), "fn", file_.BitVectorType(16, NextLoc()));
  LogicRef* a =
      fn->AddArgument(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                      file_.BitVectorType(24, NextLoc())),
                      NextLoc());
  fn->AddStatement<ReturnStatement>(NextLoc(),
                                    file_.Add(a, BareLiteral(1), NextLoc()));
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
  auto* width = module_->AddParameter("width", BareLiteral(24), NextLoc());
  Typedef* word_t = module_->AddTypedef(
      file_.Make<Def>(
          NextLoc(), "word_t", DataKind::kLogic,
          file_.Make<BitVectorType>(
              NextLoc(), file_.Sub(width, BareLiteral(1), NextLoc()),
              /*is_signed=*/false, /*size_expr_is_max=*/true)),
      NextLoc());
  TypedefType* word_t_type = file_.Make<TypedefType>(NextLoc(), word_t);
  VerilogFunction* fn =
      file_.Make<VerilogFunction>(NextLoc(), "fn", word_t_type);
  LogicRef* a = fn->AddArgument(
      file_.Make<Def>(NextLoc(), "a", DataKind::kUser, word_t_type), NextLoc());
  fn->AddStatement<ReturnStatement>(NextLoc(),
                                    file_.Add(a, UnsignedZero(3), NextLoc()));
  module_->top()->AddModuleMember(fn);
  EXPECT_EQ(InferTypesToString(),
            R"(
0 : word_t
1 : integer
24 : integer
a : word_t
a + 0 : word_t
width : integer
width - 1 : integer
)");
}

TEST_F(InferVastTypesTest, PreserveSameTypedef) {
  // typedef logic[23:0] word_t;
  // parameter word_t a = 3;
  // parameter word_t b = 4;
  // a + b
  Typedef* word_t = module_->AddTypedef(
      file_.Make<Def>(NextLoc(), "word_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(NextLoc(), BareLiteral(23),
                                                /*is_signed=*/false,
                                                /*size_expr_is_max=*/true)),
      NextLoc());
  TypedefType* word_t_type = file_.Make<TypedefType>(NextLoc(), word_t);
  auto* a = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "a", DataKind::kUser, word_t_type),
      BareLiteral(3), NextLoc());
  auto* b = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "b", DataKind::kUser, word_t_type),
      BareLiteral(4), NextLoc());
  EXPECT_EQ(InferTypesToString(file_.Add(a, b, NextLoc())),
            R"(
a : word_t
b : word_t
a + b : word_t
)");
}

TEST_F(InferVastTypesTest, GenerifyDifferentTypedefs) {
  // typedef logic[23:0] word_t;
  // typedef logic[23:0] thing_t;
  // parameter word_t a = 3;
  // parameter thing_t b = 4;
  // a + b
  // should "promote" them both to the generic type.
  Typedef* word_t = module_->AddTypedef(
      file_.Make<Def>(NextLoc(), "word_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(NextLoc(), BareLiteral(23),
                                                /*is_signed=*/false,
                                                /*size_expr_is_max=*/true)),
      NextLoc());
  Typedef* thing_t = module_->AddTypedef(
      file_.Make<Def>(NextLoc(), "thing_t", DataKind::kLogic,
                      file_.Make<BitVectorType>(NextLoc(), BareLiteral(23),
                                                /*is_signed=*/false,
                                                /*size_expr_is_max=*/true)),
      NextLoc());
  auto* a = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "a", DataKind::kUser,
                      file_.Make<TypedefType>(NextLoc(), word_t)),
      BareLiteral(3), NextLoc());
  auto* b = module_->AddParameter(
      file_.Make<Def>(NextLoc(), "b", DataKind::kUser,
                      file_.Make<TypedefType>(NextLoc(), thing_t)),
      BareLiteral(4), NextLoc());
  EXPECT_EQ(InferTypesToString(file_.Add(a, b, NextLoc())),
            R"(
a : [23:0]
b : [23:0]
a + b : [23:0]
)");
}

TEST_F(InferVastTypesTest, ContextDependentUnary) {
  // b = ~(a + 0) where all parameters are logic [15:0].
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            uint16_max_, NextLoc());
  module_->AddParameter(
      file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                      file_.BitVectorType(16, NextLoc())),
      file_.BitwiseNot(file_.Add(a, BareLiteral(0), NextLoc()), NextLoc()),
      NextLoc());
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.ScalarType(NextLoc())),
                            BareLiteral(0), NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(3, NextLoc())),
                            BareLiteral(0), NextLoc());
  auto* c =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                                            file_.BitVectorType(2, NextLoc())),
                            BareLiteral(0), NextLoc());
  EXPECT_EQ(InferTypesToString(
                file_.Add(a, file_.GreaterThan(b, c, NextLoc()), NextLoc())),
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
  auto* a =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            BareLiteral(0), NextLoc());
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            BareLiteral(0), NextLoc());
  EXPECT_EQ(InferTypesToString(file_.LogicalAnd(
                a, file_.Add(b, BareLiteral(1), NextLoc()), NextLoc())),
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
  auto* b =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "b", DataKind::kLogic,
                                            file_.BitVectorType(16, NextLoc())),
                            BareLiteral(0), NextLoc());
  auto* c =
      module_->AddParameter(file_.Make<Def>(NextLoc(), "c", DataKind::kLogic,
                                            file_.BitVectorType(32, NextLoc())),
                            BareLiteral(0), NextLoc());
  module_->AddParameter(file_.Make<Def>(NextLoc(), "a", DataKind::kLogic,
                                        file_.BitVectorType(16, NextLoc())),
                        file_.AndReduce(file_.Add(b, c, NextLoc()), NextLoc()),
                        NextLoc());
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
  auto* foo = module_->AddParameter("foo", UnsignedZero(4), NextLoc());
  module_->AddParameter("bar", file_.Add(UnsignedZero(2), foo, NextLoc()),
                        NextLoc());
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
