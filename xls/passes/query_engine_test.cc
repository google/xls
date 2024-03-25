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

#include "xls/passes/query_engine.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/ternary.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::IsOk;
using testing::Not;

enum class QueryEngineType { kTernary, kBdd };

// A test of the query engine parameterized to test both ternary and BDD based
// engines. Tests basic functionality common to both engines.
class QueryEngineTest : public IrTestBase,
                        public testing::WithParamInterface<QueryEngineType> {
 protected:
  absl::StatusOr<std::unique_ptr<QueryEngine>> GetEngine(Function* f) {
    if (GetParam() == QueryEngineType::kTernary) {
      std::unique_ptr<TernaryQueryEngine> engine =
          std::make_unique<TernaryQueryEngine>();
      XLS_RETURN_IF_ERROR(engine->Populate(f).status());
      return engine;
    }
    if (GetParam() == QueryEngineType::kBdd) {
      std::unique_ptr<BddQueryEngine> engine =
          std::make_unique<BddQueryEngine>();
      XLS_RETURN_IF_ERROR(engine->Populate(f).status());
      return engine;
    }
    LOG(FATAL) << "Update QueryEngineTest::GetEngine to match QueryEngineType";
  }

  // Create a BValue with known bits equal to the given ternary vector. Created
  // using a param and AND/OR masks.
  BValue MakeValueWithKnownBits(std::string_view name,
                                const TernaryVector& known_bits,
                                FunctionBuilder* fb) {
    absl::InlinedVector<bool, 1> known_zeros;
    absl::InlinedVector<bool, 1> known_ones;
    for (TernaryValue value : known_bits) {
      known_zeros.push_back(value == TernaryValue::kKnownZero);
      known_ones.push_back(value == TernaryValue::kKnownOne);
    }
    BValue and_mask = fb->Literal(bits_ops::Not(Bits(known_zeros)));
    BValue or_mask = fb->Literal(Bits(known_ones));
    return fb->Or(or_mask,
                  fb->And(and_mask, fb->Param(name, fb->package()->GetBitsType(
                                                        known_bits.size()))));
  }

  // Runs QueryEngine on the op created with the passed in function. The
  // input to the op is crafted to have known bits equal to the given
  // TernaryVector.
  absl::StatusOr<std::string> RunOnUnaryOp(
      std::string_view operand_known_bits,
      std::function<void(BValue, FunctionBuilder*)> make_op) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue operand = MakeValueWithKnownBits(
        "input", StringToTernaryVector(operand_known_bits).value(), &fb);
    make_op(operand, &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
    return engine->ToString(f->return_value());
  }

  // Runs QueryEngine on the op created with the passed in function. The
  // inputs to the op is crafted to have known bits equal to the given
  // TernaryVectors.
  absl::StatusOr<std::string> RunOnBinaryOp(
      std::string_view lhs_known_bits, std::string_view rhs_known_bits,
      std::function<void(BValue, BValue, FunctionBuilder*)> make_op) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue lhs = MakeValueWithKnownBits(
        "lhs", StringToTernaryVector(lhs_known_bits).value(), &fb);
    BValue rhs = MakeValueWithKnownBits(
        "rhs", StringToTernaryVector(rhs_known_bits).value(), &fb);
    make_op(lhs, rhs, &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
    return engine->ToString(f->return_value());
  }

  absl::StatusOr<std::string> GetMaxUnsignedValue(std::string_view known_bits) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue n = MakeValueWithKnownBits(
        "value", StringToTernaryVector(known_bits).value(), &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
    return absl::StrCat("0b",
                        BitsToRawDigits(engine->MaxUnsignedValue(n.node()),
                                        FormatPreference::kBinary,
                                        /*emit_leading_zeros=*/true));
  }

  absl::StatusOr<std::string> GetMinUnsignedValue(std::string_view known_bits) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue n = MakeValueWithKnownBits(
        "value", StringToTernaryVector(known_bits).value(), &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
    return absl::StrCat("0b",
                        BitsToRawDigits(engine->MinUnsignedValue(n.node()),
                                        FormatPreference::kBinary,
                                        /*emit_leading_zeros=*/true));
  }

  absl::StatusOr<bool> GetNodesKnownUnsignedNotEquals(
      std::string_view lhs_known_bits, std::string_view rhs_known_bits) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue lhs = MakeValueWithKnownBits(
        "lhs", StringToTernaryVector(lhs_known_bits).value(), &fb);
    BValue rhs = MakeValueWithKnownBits(
        "rhs", StringToTernaryVector(rhs_known_bits).value(), &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
    return engine->NodesKnownUnsignedNotEquals(lhs.node(), rhs.node());
  }

  absl::StatusOr<bool> GetNodesKnownUnsignedEquals(
      std::string_view lhs_known_bits, std::string_view rhs_known_bits) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue lhs = MakeValueWithKnownBits(
        "lhs", StringToTernaryVector(lhs_known_bits).value(), &fb);
    BValue rhs = MakeValueWithKnownBits(
        "rhs", StringToTernaryVector(rhs_known_bits).value(), &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
    return engine->NodesKnownUnsignedEquals(lhs.node(), rhs.node());
  }
};

TEST_P(QueryEngineTest, SimpleBinaryOp) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[32], y: bits[32]) -> bits[32] {
        ret add.1: bits[32] = add(x, y)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  for (int64_t i = 0; i < 32; ++i) {
    for (Node* node :
         {FindNode("x", f), FindNode("y", f), FindNode("add.1", f)}) {
      EXPECT_FALSE(engine->IsKnown(TreeBitLocation(node, i)));
      EXPECT_FALSE(engine->IsOne(TreeBitLocation(node, i)));
      EXPECT_FALSE(engine->IsZero(TreeBitLocation(node, i)));
    }
  }
}

TEST_P(QueryEngineTest, OneLiteral) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f() -> bits[16] {
        ret literal.1: bits[16] = literal(value=0x0ff0)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  ASSERT_TRUE(engine->IsTracked(f->return_value()));
  EXPECT_EQ(engine->ToString(f->return_value()), "0b0000_1111_1111_0000");
  EXPECT_FALSE(engine->IsAllOnes(f->return_value()));
  EXPECT_FALSE(engine->IsAllZeros(f->return_value()));
  EXPECT_EQ(engine->ToString(f->return_value()), "0b0000_1111_1111_0000");
}

TEST_P(QueryEngineTest, BitSlice) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f() -> bits[9] {
        literal.1: bits[16] = literal(value=0x0ff0)
        ret bit_slice.2: bits[9] = bit_slice(literal.1, start=5, width=9)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  ASSERT_TRUE(engine->IsTracked(f->return_value()));
  EXPECT_EQ(engine->ToString(f->return_value()), "0b0_0111_1111");
}

TEST_P(QueryEngineTest, OneHotLsbToMsb) {
  auto make_one_hot = [](BValue operand, FunctionBuilder* fb) {
    fb->OneHot(operand, LsbOrMsb::kLsb);
  };
  EXPECT_THAT(RunOnUnaryOp("0bXX1", make_one_hot), IsOkAndHolds("0b0001"));
  EXPECT_THAT(RunOnUnaryOp("0bX1X", make_one_hot), IsOkAndHolds("0b00XX"));
  EXPECT_THAT(RunOnUnaryOp("0bX0X", make_one_hot), IsOkAndHolds("0bXX0X"));
}

TEST_P(QueryEngineTest, OneHotMsbToLsb) {
  auto make_one_hot = [](BValue operand, FunctionBuilder* fb) {
    fb->OneHot(operand, LsbOrMsb::kMsb);
  };
  EXPECT_THAT(RunOnUnaryOp("0bXX1", make_one_hot), IsOkAndHolds("0b0XXX"));
  EXPECT_THAT(RunOnUnaryOp("0bX1X", make_one_hot), IsOkAndHolds("0b0XX0"));
  EXPECT_THAT(RunOnUnaryOp("0bX0X", make_one_hot), IsOkAndHolds("0bXX0X"));
}

TEST_P(QueryEngineTest, OneHotSelect) {
  {
    auto make_select = [](BValue operand, FunctionBuilder* fb) {
      std::vector<BValue> cases{fb->Param("foo", fb->package()->GetBitsType(4)),
                                fb->Literal(UBits(0b0011, 4)),
                                fb->Literal(UBits(0b0101, 4))};
      fb->OneHotSelect(operand, cases);
    };
    EXPECT_THAT(RunOnUnaryOp("0bXX1", make_select), IsOkAndHolds("0bXXXX"));
    EXPECT_THAT(RunOnUnaryOp("0bX1X", make_select), IsOkAndHolds("0bXX11"));
    EXPECT_THAT(RunOnUnaryOp("0b1XX", make_select), IsOkAndHolds("0bX1X1"));
    EXPECT_THAT(RunOnUnaryOp("0b0X0", make_select), IsOkAndHolds("0b00XX"));
  }
}

TEST_P(QueryEngineTest, OneHotSelectPrecededByOneHot) {
  {
    auto make_select = [](BValue operand, FunctionBuilder* fb) {
      std::vector<BValue> cases{fb->Param("foo", fb->package()->GetBitsType(4)),
                                fb->Literal(UBits(0b0011, 4)),
                                fb->Literal(UBits(0b0101, 4))};
      fb->OneHotSelect(fb->OneHot(operand, LsbOrMsb::kLsb), cases);
    };
    EXPECT_THAT(RunOnUnaryOp("0bX1", make_select), IsOkAndHolds("0bXXXX"));
    EXPECT_THAT(RunOnUnaryOp("0b1X", make_select), IsOkAndHolds("0bXXXX"));
    EXPECT_THAT(RunOnUnaryOp("0bX0", make_select), IsOkAndHolds("0b0XX1"));
    EXPECT_THAT(RunOnUnaryOp("0b0X", make_select), IsOkAndHolds("0bXXXX"));
  }
}

TEST_P(QueryEngineTest, Shll) {
  // TODO(meheff): Enable test for BDD query engine when shifts are supported.
  if (GetParam() == QueryEngineType::kBdd) {
    return;
  }
  auto make_shll = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Shll(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b110", make_shll),
              IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0b111", make_shll),
              IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0bX1X", make_shll),
              IsOkAndHolds("0bX00"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0bXXX", make_shll),
              IsOkAndHolds("0bXXX"));
}

TEST_P(QueryEngineTest, Shra) {
  // TODO(meheff): Enable test for BDD query engine when shifts are supported.
  if (GetParam() == QueryEngineType::kBdd) {
    return;
  }
  auto make_shra = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Shra(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b110", make_shra),
              IsOkAndHolds("0bXXX"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0b111", make_shra),
              IsOkAndHolds("0b111"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0bX1X", make_shra),
              IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0bXXX", make_shra),
              IsOkAndHolds("0b0XX"));
}

TEST_P(QueryEngineTest, Shrl) {
  // TODO(meheff): Enable test for BDD query engine when shifts are supported.
  if (GetParam() == QueryEngineType::kBdd) {
    return;
  }
  auto make_shrl = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Shrl(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b110", make_shrl),
              IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_shrl),
              IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0bX1X", make_shrl),
              IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0bXXX", make_shrl),
              IsOkAndHolds("0b0XX"));
}

TEST_P(QueryEngineTest, Concat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[3], y: bits[4]) -> bits[16] {
        literal.1: bits[4] = literal(value=0b1100)
        literal.2: bits[5] = literal(value=0b10101)
        ret concat.3: bits[16] = concat(x, literal.1, y, literal.2)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(f->return_value()), "0bXXX1_100X_XXX1_0101");
}

TEST_P(QueryEngineTest, BitSliceOfConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[3], y: bits[4]) -> bits[6] {
        literal.1: bits[4] = literal(value=0b1100)
        literal.2: bits[5] = literal(value=0b10101)
        concat.3: bits[16] = concat(x, literal.1, y, literal.2)
        ret bit_slice.4: bits[6] = bit_slice(concat.3, start=4, width=6)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(f->return_value()), "0b0X_XXX1");
}

TEST_P(QueryEngineTest, AllOnesOrZeros) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f() -> bits[16] {
        literal.1: bits[16] = literal(value=0)
        ret literal.2: bits[16] = literal(value=0xffff)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_FALSE(engine->IsAllOnes(FindNode("literal.1", f)));
  EXPECT_TRUE(engine->IsAllZeros(FindNode("literal.1", f)));
  EXPECT_TRUE(engine->IsAllOnes(FindNode("literal.2", f)));
  EXPECT_FALSE(engine->IsAllZeros(FindNode("literal.2", f)));
}

TEST_P(QueryEngineTest, BitwiseOps) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[8]) -> bits[24] {
        literal.1: bits[16] = literal(value=0xff00)
        literal.2: bits[16] = literal(value=0xf0f0)
        concat.3: bits[24] = concat(x, literal.1)
        concat.4: bits[24] = concat(y, literal.2)
        and.5: bits[24] = and(concat.3, concat.4)
        not.6: bits[24] = not(concat.3)
        or.7: bits[24] = or(concat.3, concat.4)
        ret xor.8: bits[24] = xor(concat.3, concat.4)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("and.5", f)),
            "0bXXXX_XXXX_1111_0000_0000_0000");
  EXPECT_EQ(engine->ToString(FindNode("not.6", f)),
            "0bXXXX_XXXX_0000_0000_1111_1111");
  EXPECT_EQ(engine->ToString(FindNode("or.7", f)),
            "0bXXXX_XXXX_1111_1111_1111_0000");
  EXPECT_EQ(engine->ToString(FindNode("xor.8", f)),
            "0bXXXX_XXXX_0000_1111_1111_0000");
}

TEST_P(QueryEngineTest, SignExtend) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[4]) -> bits[24] {
        literal.1: bits[4] = literal(value=0xa)
        literal.2: bits[4] = literal(value=0x4)
        concat.3: bits[8] = concat(literal.1, y)
        concat.4: bits[8] = concat(literal.2, y)

        // Known sign. Sign bit is one.
        sign_ext.5: bits[16] = sign_ext(concat.3, new_bit_count=16)

        // Known sign. Sign bit is zero.
        sign_ext.6: bits[16] = sign_ext(concat.4, new_bit_count=16)

        // Identity sign-extension.
        sign_ext.7: bits[8] = sign_ext(concat.3, new_bit_count=8)

        // Unknown sign.
        concat.8: bits[16] = concat(x, concat.3)
        ret sign_ext.9: bits[24] = sign_ext(concat.8, new_bit_count=24)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("sign_ext.5", f)),
            "0b1111_1111_1010_XXXX");
  EXPECT_EQ(engine->ToString(FindNode("sign_ext.6", f)),
            "0b0000_0000_0100_XXXX");
  EXPECT_EQ(engine->ToString(FindNode("sign_ext.7", f)), "0b1010_XXXX");
  EXPECT_EQ(engine->ToString(FindNode("sign_ext.9", f)),
            "0bXXXX_XXXX_XXXX_XXXX_1010_XXXX");
}

TEST_P(QueryEngineTest, BinarySelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[1], y: bits[8]) -> bits[24] {
        literal.1: bits[16] = literal(value=0xff00)
        literal.2: bits[16] = literal(value=0xf0f0)
        concat.3: bits[24] = concat(literal.1, y)
        concat.4: bits[24] = concat(literal.2, y)

        ret sel.5: bits[24] = sel(p, cases=[concat.3, concat.4])
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("sel.5", f)),
            "0b1111_XXXX_XXXX_0000_XXXX_XXXX");
}

TEST_P(QueryEngineTest, TernarySelectWithDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(p: bits[2]) -> bits[32] {
        literal.1: bits[32] = literal(value=0xfffff000)
        literal.2: bits[32] = literal(value=0xff00ff00)
        literal.3: bits[32] = literal(value=0xf0f0f0f0)
        literal.4: bits[32] = literal(value=0x0ffffff0)
        ret sel.5: bits[32] = sel(p, cases=[literal.1, literal.2, literal.3], default=literal.4)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("sel.5", f)),
            "0bXXXX_XXXX_XXXX_XXXX_1111_XXXX_XXXX_0000");
}

TEST_P(QueryEngineTest, AndWithHighBitZeroes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8], y: bits[16]) -> bits[16] {
        literal.2: bits[8] = literal(value=0)
        concat.3: bits[16] = concat(literal.2, x)
        ret and.4: bits[16] = and(concat.3, y)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("and.4", f)), "0b0000_0000_XXXX_XXXX");
}

TEST_P(QueryEngineTest, NandTruthTable) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f() -> bits[4] {
        literal.1: bits[4] = literal(value=0b0101)
        literal.2: bits[4] = literal(value=0b0011)
        ret nand.3: bits[4] = nand(literal.1, literal.2)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("nand.3", f)), "0b1110");
}

TEST_P(QueryEngineTest, NorTruthTable) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f() -> bits[4] {
        literal.1: bits[4] = literal(value=0b0101)
        literal.2: bits[4] = literal(value=0b0011)
        ret nor.3: bits[4] = nor(literal.1, literal.2)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("nor.3", f)), "0b1000");
}

TEST_P(QueryEngineTest, ShrlFullyKnownRhs) {
  // TODO(meheff): Enable test for BDD query engine when shifts are supported.
  if (GetParam() == QueryEngineType::kBdd) {
    return;
  }
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8]) -> bits[16] {
        literal.2: bits[8] = literal(value=0xee)
        concat.3: bits[16] = concat(literal.2, x)
        literal.4: bits[16] = literal(value=8)
        ret shrl.5: bits[16] = shrl(concat.3, literal.4)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_EQ(engine->ToString(FindNode("concat.3", f)), "0b1110_1110_XXXX_XXXX");
  EXPECT_EQ(engine->ToString(FindNode("shrl.5", f)), "0b0000_0000_1110_1110");
}

TEST_P(QueryEngineTest, ShrlPartiallyKnownRhs) {
  // TODO(meheff): Enable test for BDD query engine when shifts are supported.
  if (GetParam() == QueryEngineType::kBdd) {
    return;
  }
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn f(x: bits[8]) -> bits[8] {
        literal.2: bits[8] = literal(value=0xff)
        literal.3: bits[8] = literal(value=1)
        or.4: bits[8] = or(x, literal.3)
        ret shrl.5: bits[8] = shrl(x, or.4)
     }
  )",
                                                       p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  // The shift amount is at least 0b1 so we know the high bit must be zero.
  EXPECT_EQ(engine->ToString(FindNode("shrl.5", f)), "0b0XXX_XXXX");
}

TEST_P(QueryEngineTest, Decode) {
  auto make_decode = [](BValue operand, FunctionBuilder* fb) {
    fb->Decode(operand, /*width=*/8);
  };
  EXPECT_THAT(RunOnUnaryOp("0b000", make_decode), IsOkAndHolds("0b0000_0001"));
  EXPECT_THAT(RunOnUnaryOp("0b001", make_decode), IsOkAndHolds("0b0000_0010"));
  EXPECT_THAT(RunOnUnaryOp("0b101", make_decode), IsOkAndHolds("0b0010_0000"));
  EXPECT_THAT(RunOnUnaryOp("0bXX1", make_decode), IsOkAndHolds("0bX0X0_X0X0"));
  EXPECT_THAT(RunOnUnaryOp("0bX1X", make_decode), IsOkAndHolds("0bXX00_XX00"));
  EXPECT_THAT(RunOnUnaryOp("0bX0X", make_decode), IsOkAndHolds("0b00XX_00XX"));
}

TEST_P(QueryEngineTest, Encode) {
  auto make_encode = [](BValue operand, FunctionBuilder* fb) {
    fb->Encode(operand);
  };
  EXPECT_THAT(RunOnUnaryOp("0b0000", make_encode), IsOkAndHolds("0b00"));
  EXPECT_THAT(RunOnUnaryOp("0b0010", make_encode), IsOkAndHolds("0b01"));
  EXPECT_THAT(RunOnUnaryOp("0b1010", make_encode), IsOkAndHolds("0b11"));
  EXPECT_THAT(RunOnUnaryOp("0bXX10", make_encode), IsOkAndHolds("0bX1"));
  EXPECT_THAT(RunOnUnaryOp("0bX1X0", make_encode), IsOkAndHolds("0b1X"));
  EXPECT_THAT(RunOnUnaryOp("0bX0X0", make_encode), IsOkAndHolds("0bXX"));
}

TEST_P(QueryEngineTest, Reverse) {
  auto make_reverse = [](BValue operand, FunctionBuilder* fb) {
    fb->Reverse(operand);
  };
  EXPECT_THAT(RunOnUnaryOp("0b0000", make_reverse), IsOkAndHolds("0b0000"));
  EXPECT_THAT(RunOnUnaryOp("0b0010", make_reverse), IsOkAndHolds("0b0100"));
  EXPECT_THAT(RunOnUnaryOp("0b1010", make_reverse), IsOkAndHolds("0b0101"));
  EXPECT_THAT(RunOnUnaryOp("0bXX10", make_reverse), IsOkAndHolds("0b01XX"));
  EXPECT_THAT(RunOnUnaryOp("0bX1X0", make_reverse), IsOkAndHolds("0b0X1X"));
  EXPECT_THAT(RunOnUnaryOp("0bX0X0", make_reverse), IsOkAndHolds("0b0X0X"));
}

TEST_P(QueryEngineTest, MaxUnsignedValue) {
  EXPECT_THAT(GetMaxUnsignedValue("0b"), IsOkAndHolds("0b0"));
  EXPECT_THAT(GetMaxUnsignedValue("0b0"), IsOkAndHolds("0b0"));
  EXPECT_THAT(GetMaxUnsignedValue("0b1"), IsOkAndHolds("0b1"));
  EXPECT_THAT(GetMaxUnsignedValue("0bX"), IsOkAndHolds("0b1"));
  EXPECT_THAT(GetMaxUnsignedValue("0b0000"), IsOkAndHolds("0b0000"));
  EXPECT_THAT(GetMaxUnsignedValue("0b1111"), IsOkAndHolds("0b1111"));
  EXPECT_THAT(GetMaxUnsignedValue("0b0101"), IsOkAndHolds("0b0101"));
  EXPECT_THAT(GetMaxUnsignedValue("0b1010"), IsOkAndHolds("0b1010"));
  EXPECT_THAT(GetMaxUnsignedValue("0bXX10"), IsOkAndHolds("0b1110"));
  EXPECT_THAT(GetMaxUnsignedValue("0b10XX"), IsOkAndHolds("0b1011"));
  EXPECT_THAT(GetMaxUnsignedValue("0bXXXX"), IsOkAndHolds("0b1111"));
}

TEST_P(QueryEngineTest, MinUnsignedValue) {
  EXPECT_THAT(GetMinUnsignedValue("0b"), IsOkAndHolds("0b0"));
  EXPECT_THAT(GetMinUnsignedValue("0b0"), IsOkAndHolds("0b0"));
  EXPECT_THAT(GetMinUnsignedValue("0b1"), IsOkAndHolds("0b1"));
  EXPECT_THAT(GetMinUnsignedValue("0bX"), IsOkAndHolds("0b0"));
  EXPECT_THAT(GetMinUnsignedValue("0b0000"), IsOkAndHolds("0b0000"));
  EXPECT_THAT(GetMinUnsignedValue("0b1111"), IsOkAndHolds("0b1111"));
  EXPECT_THAT(GetMinUnsignedValue("0b0101"), IsOkAndHolds("0b0101"));
  EXPECT_THAT(GetMinUnsignedValue("0b1010"), IsOkAndHolds("0b1010"));
  EXPECT_THAT(GetMinUnsignedValue("0bXX10"), IsOkAndHolds("0b0010"));
  EXPECT_THAT(GetMinUnsignedValue("0b10XX"), IsOkAndHolds("0b1000"));
  EXPECT_THAT(GetMinUnsignedValue("0bXXXX"), IsOkAndHolds("0b0000"));
}

TEST_P(QueryEngineTest, NodesKnownUnsignedNotEquals) {
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b", "0b"), IsOkAndHolds(false));

  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b0", "0b0"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b0", "0b1"), IsOkAndHolds(true));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b0", "0bX"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b1", "0bX"),
              IsOkAndHolds(false));

  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b00X0", "0b0010"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b00X0", "0b1010"),
              IsOkAndHolds(true));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0bXXX0", "0b1010"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0bXXX0", "0b1011"),
              IsOkAndHolds(true));

  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0bXXXX", "0b11"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0bX1XX", "0b11"),
              IsOkAndHolds(true));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0bX0XX", "0b11"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedNotEquals("0b10XX", "0bXX"),
              IsOkAndHolds(true));
}

TEST_P(QueryEngineTest, NodesKnownUnsignedEquals) {
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b", "0b"), IsOkAndHolds(true));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b", "0b00000"), IsOkAndHolds(true));

  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b0", "0b0"), IsOkAndHolds(true));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b0", "0b1"), IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b0", "0bX"), IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b1", "0bX"), IsOkAndHolds(false));

  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b00X0", "0b0010"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b00X0", "0b1010"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0bXXX0", "0b1010"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0bXXX0", "0b1011"),
              IsOkAndHolds(false));

  EXPECT_THAT(GetNodesKnownUnsignedEquals("0bXXXX", "0b11"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0bXX11", "0b11"),
              IsOkAndHolds(false));
  EXPECT_THAT(GetNodesKnownUnsignedEquals("0b0011", "0b11"),
              IsOkAndHolds(true));

  // Verify that a node is equal to itself even if nothing is known about the
  // node's value.
  Package p("test_package");
  FunctionBuilder fb("f", &p);
  BValue a = fb.Param("a", p.GetBitsType(42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));
  EXPECT_TRUE(engine->NodesKnownUnsignedEquals(a.node(), a.node()));
}

TEST_P(QueryEngineTest, DefaultSpecializeDoesNothing) {
  if (GetParam() != QueryEngineType::kTernary &&
      GetParam() != QueryEngineType::kBdd) {
    // Only these two should not have any specializations.
    return;
  }
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // fn (x: u8, y: u8) { if (x < 4:u8) { x + 4 } else { y + 4 } }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue x_plus_4 = fb.Add(x, fb.Literal(UBits(4, 8)));
  BValue y_plus_4 = fb.Add(y, fb.Literal(UBits(4, 8)));
  BValue cmp = fb.ULt(x, fb.Literal(UBits(4, 8)));
  BValue result = fb.Select(cmp, {y_plus_4, x_plus_4});

  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.BuildWithReturnValue(result));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> engine, GetEngine(f));

  auto consequent_engine = engine->SpecializeGivenPredicate(
      {PredicateState(result.node()->As<Select>(), 1)});
  auto alternate_engine = engine->SpecializeGivenPredicate(
      {PredicateState(result.node()->As<Select>(), 0)});
  auto secondary_consequent_engine =
      consequent_engine->SpecializeGivenPredicate({});

  EXPECT_THAT(consequent_engine->Populate(f), Not(IsOk()));
  EXPECT_THAT(alternate_engine->Populate(f), Not(IsOk()));

#define EXPECT_EQUIV(call)                                    \
  EXPECT_EQ(consequent_engine->call, engine->call);           \
  EXPECT_EQ(secondary_consequent_engine->call, engine->call); \
  EXPECT_EQ(alternate_engine->call, engine->call)

  EXPECT_EQUIV(IsTracked(cmp.node()));
  EXPECT_EQUIV(AtLeastOneNodeTrue({cmp.node()}));
  EXPECT_EQUIV(AtMostOneNodeTrue({cmp.node()}));
  EXPECT_EQUIV(AtMostOneBitTrue(cmp.node()));
  EXPECT_EQUIV(AtLeastOneBitTrue(cmp.node()));
  EXPECT_EQUIV(GetTernary(x_plus_4.node()));
  EXPECT_EQUIV(GetIntervals(x_plus_4.node()));
  EXPECT_EQUIV(AtMostOneTrue({TreeBitLocation(x_plus_4.node(), 1),
                              TreeBitLocation(x_plus_4.node(), 7)}));
  EXPECT_EQUIV(AtLeastOneTrue({TreeBitLocation(x_plus_4.node(), 1),
                               TreeBitLocation(x_plus_4.node(), 7)}));
  EXPECT_EQUIV(Implies(TreeBitLocation(cmp.node(), 0),
                       TreeBitLocation(x_plus_4.node(), 7)));
  EXPECT_EQUIV(ImpliedNodeValue({{TreeBitLocation(cmp.node(), 0), true}},
                                x_plus_4.node()));
  EXPECT_EQUIV(KnownEquals(TreeBitLocation(x_plus_4.node(), 6),
                           TreeBitLocation(x_plus_4.node(), 7)));
  EXPECT_EQUIV(KnownNotEquals(TreeBitLocation(x_plus_4.node(), 6),
                              TreeBitLocation(x_plus_4.node(), 7)));
#undef EXPECT_EQUIV
}

INSTANTIATE_TEST_SUITE_P(
    QueryEngineTestInstantiation, QueryEngineTest,
    testing::Values(QueryEngineType::kTernary, QueryEngineType::kBdd),
    [](const testing::TestParamInfo<QueryEngineTest::ParamType>& info) {
      return info.param == QueryEngineType::kTernary ? "ternary" : "bdd";
    });

}  // namespace
}  // namespace xls
