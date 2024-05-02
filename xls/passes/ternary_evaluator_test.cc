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

#include "xls/passes/ternary_evaluator.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::ElementsAre;

class TernaryLogicTest : public ::testing::Test {
 protected:
  TernaryLogicTest() = default;

  TernaryVector FromString(std::string_view s) {
    return StringToTernaryVector(s).value();
  }

  // Returns all TernaryVectors of the given width. For example, all
  // TernaryVectors of width 2 are: 0b00, 0b01, 0b0X, 0b10, 0b11, 0b1X, 0bX0,
  // 0bX1, 0bXX.
  std::vector<TernaryVector> EnumerateTernaryVectors(int64_t width) {
    std::vector<TernaryVector> vectors;
    EnumerateTernaryVectorsHelper({}, width, &vectors);
    return vectors;
  }

  // Returns all Bits objects which match the pattern of the given
  // TernaryVector. For example, TernaryVector 0b1xx0 produces the following
  // Bits values: 0b1000, 0b1010, 0b1100, 0b1110.
  std::vector<Bits> ExpandToBits(const TernaryVector& vector) {
    std::vector<Bits> result;
    ExpandToBitsHelper(vector, Bits(), &result);
    return result;
  }

  // Returns a TernaryVector which matches the given set of Bits values. For
  // each bit index, if the bit is one for all Bits values, the resulting
  // ternary value in the TernaryVector is TernaryValue::kKnownOne. If the bit
  // is zero for all Bits values, the ternary value is TernaryValue::kKnownOne.
  // Otherwise it is TernaryValue::kUnknown. Example: { 0b1000, 0b1100, 0b1001 }
  // => 0b1X0X
  TernaryVector ReduceFromBits(absl::Span<const Bits> bits_vector) {
    CHECK(!bits_vector.empty());
    TernaryVector result = evaluator_.BitsToVector(bits_vector.front());
    for (const Bits& bits : bits_vector.subspan(1)) {
      CHECK_EQ(bits.bit_count(), result.size());
      for (int64_t i = 0; i < result.size(); ++i) {
        bool same = ((bits.Get(i) && result[i] == TernaryValue::kKnownOne) ||
                     (!bits.Get(i) && result[i] == TernaryValue::kKnownZero));
        result[i] = same ? result[i] : TernaryValue::kUnknown;
      }
    }
    return result;
  }

  TernaryEvaluator evaluator_;

 private:
  void EnumerateTernaryVectorsHelper(const TernaryVector& prefix,
                                     int64_t remaining,
                                     std::vector<TernaryVector>* vectors) {
    if (remaining == 0) {
      vectors->push_back(prefix);
      return;
    }

    for (TernaryValue state :
         {TernaryValue::kKnownZero, TernaryValue::kKnownOne,
          TernaryValue::kUnknown}) {
      TernaryVector new_prefix = prefix;
      new_prefix.push_back(state);
      EnumerateTernaryVectorsHelper(new_prefix, remaining - 1, vectors);
    }
  }

  void ExpandToBitsHelper(const TernaryVector& vector, const Bits& prefix,
                          std::vector<Bits>* bits) {
    int64_t index = vector.size() - prefix.bit_count() - 1;
    if (index == -1) {
      bits->push_back(prefix);
      return;
    }
    if (vector[index] != TernaryValue::kKnownOne) {
      ExpandToBitsHelper(vector, bits_ops::Concat({prefix, UBits(0, 1)}), bits);
    }
    if (vector[index] != TernaryValue::kKnownZero) {
      ExpandToBitsHelper(vector, bits_ops::Concat({prefix, UBits(1, 1)}), bits);
    }
  }
};

TEST_F(TernaryLogicTest, StringToTernaryVector) {
  XLS_ASSERT_OK_AND_ASSIGN(TernaryVector vec, StringToTernaryVector("0b0x11X"));
  EXPECT_EQ(ToString(vec), "0b0_X11X");
  EXPECT_EQ(vec.size(), 5);
  EXPECT_EQ(vec[0], TernaryValue::kUnknown);
  EXPECT_EQ(vec[1], TernaryValue::kKnownOne);
  EXPECT_EQ(vec[2], TernaryValue::kKnownOne);
  EXPECT_EQ(vec[3], TernaryValue::kUnknown);
  EXPECT_EQ(vec[4], TernaryValue::kKnownZero);

  XLS_ASSERT_OK_AND_ASSIGN(TernaryVector empty_vec,
                           StringToTernaryVector("0b"));
  EXPECT_TRUE(empty_vec.empty());

  EXPECT_THAT(StringToTernaryVector("1x010"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StringToTernaryVector("0b123"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TernaryLogicTest, BitsToVector) {
  EXPECT_EQ(ToString(evaluator_.BitsToVector(UBits(0b110010, 6))), "0b11_0010");
  EXPECT_EQ(ToString(evaluator_.BitsToVector(UBits(0, 6))), "0b00_0000");
  EXPECT_EQ(ToString(evaluator_.BitsToVector(UBits(0, 0))), "0b");
}

TEST_F(TernaryLogicTest, BitwiseOps) {
  EXPECT_EQ(ToString(evaluator_.BitwiseNot(FromString("0b1X0"))), "0b0X1");
  EXPECT_EQ(ToString(evaluator_.BitwiseAnd(FromString("0b111XXX000"),
                                           FromString("0b0X10X10X1"))),
            "0b0_X10X_X000");
  // N-ary and with two operands.
  EXPECT_EQ(ToString(evaluator_.BitwiseAnd(
                {FromString("0b111XXX000"), FromString("0b0X10X10X1")})),
            "0b0_X10X_X000");
  // N-ary and with three operands.
  EXPECT_EQ(ToString(evaluator_.BitwiseAnd({FromString("0b111XXX000"),
                                            FromString("0b0X10X10X1"),
                                            FromString("0b010001010")})),
            "0b0_X000_X000");
  // N-ary and with one operands.
  EXPECT_EQ(ToString(evaluator_.BitwiseAnd({FromString("0b111XXX000")})),
            "0b1_11XX_X000");

  EXPECT_EQ(ToString(evaluator_.BitwiseOr(FromString("0b111XXX000"),
                                          FromString("0b0X10X10X1"))),
            "0b1_11XX_10X1");

  EXPECT_EQ(ToString(evaluator_.BitwiseXor(FromString("0b111XXX000"),
                                           FromString("0b0X10X10X1"))),
            "0b1_X0XX_X0X1");
}

TEST_F(TernaryLogicTest, Equals) {
  EXPECT_EQ(evaluator_.Equals(FromString("0b101"), FromString("0bXXX")),
            TernaryValue::kUnknown);
  EXPECT_EQ(evaluator_.Equals(FromString("0b101"), FromString("0bX0X")),
            TernaryValue::kUnknown);
  EXPECT_EQ(evaluator_.Equals(FromString("0b101"), FromString("0bX1X")),
            TernaryValue::kKnownZero);
  EXPECT_EQ(evaluator_.Equals(FromString("0bX01"), FromString("0bX01")),
            TernaryValue::kUnknown);
  EXPECT_EQ(evaluator_.Equals(FromString("0b1101"), FromString("0b1101")),
            TernaryValue::kKnownOne);
  EXPECT_EQ(evaluator_.Equals(FromString("0b0101"), FromString("0b1101")),
            TernaryValue::kKnownZero);
}

TEST_F(TernaryLogicTest, BitSlice) {
  EXPECT_EQ(ToString(evaluator_.BitSlice(FromString("0b11XX00"), 0, 6)),
            "0b11_XX00");
  EXPECT_EQ(ToString(evaluator_.BitSlice(FromString("0b11XX00"), 0, 0)), "0b");
  EXPECT_EQ(ToString(evaluator_.BitSlice(FromString("0b11XX00"), 3, 0)), "0b");
  EXPECT_EQ(ToString(evaluator_.BitSlice(FromString("0b11XX00"), 2, 3)),
            "0b1XX");
}
TEST_F(TernaryLogicTest, Concat) {
  EXPECT_EQ(ToString(evaluator_.Concat({FromString("0b11XX00")})), "0b11_XX00");
  EXPECT_EQ(
      ToString(evaluator_.Concat({FromString("0b1X0"), FromString("0b11")})),
      "0b1_X011");
  EXPECT_EQ(
      ToString(evaluator_.Concat({FromString("0bX"), FromString("0b1"),
                                  FromString("0b"), FromString("0b101")})),
      "0bX_1101");
}

// Because of the complexity of reasoning about ternary logic the complex ops
// (e.g., ULessThan and OneHotSelect) are exhaustively tested with non-trivial
// inputs. In each case, all possible ternary vector inputs of a fixed width are
// generated and for each ternary input the operation is evaluated to produce a
// ternary result. To check the ternary result, all concrete Bits values
// matching the ternary inputs are generated and the operation is evaluated to
// produce a set of concrete Bits results. These results are checked against the
// constraints implied by the ternary result and that the ternary result is as
// constrained as possible (e.g., if the Bits results all have one in a
// particular position, the ternary result should have a TernaryValue::kKnownOne
// in that position).
//
// For example, consider exhaustively evaluating a 2-bit-wide operation Foo. All
// nine possible ternary inputs are evaluated:
//
//   { 0b00, 0b01, 0b0X, 0b10, 0b11, 0b1X, 0bX0, 0bX1, 0bXX }
//
// For each ternary input the operation is evaluated. For example, for input
// 0b0X we may have:
//
//   Foo(0b0X) => 0bX1
//
// To verify this result (0bX1), all concrete Bits inputs which match the
// ternary input are generated. In this case:
//
//   0b0X expands to: { Bits(0b00), Bits(0b01) }
//
// And the operation is applied to each concrete Bits input:
//
//   Foo(0b00) => 0b01
//   Foo(0b01) => 0b11
//
// The set of concrete results (0b01 and 0b11) should satisfy the constraints
// implied by the ternary result (0bX1). Also the ternary result should be
// maximally constrained. That is, the ternary result should not have an 'X'
// value in the position if the concrete results always has a '1' (or always has
// a '0').
TEST_F(TernaryLogicTest, ULessThan) {
  // Enumerate all 3-wide ternary inputs.
  for (const TernaryVector& lhs : EnumerateTernaryVectors(/*width=*/3)) {
    for (const TernaryVector& rhs : EnumerateTernaryVectors(/*width=*/3)) {
      std::vector<Bits> results;
      for (const Bits& lhs_bits : ExpandToBits(lhs)) {
        for (const Bits& rhs_bits : ExpandToBits(rhs)) {
          results.push_back(UBits(
              static_cast<uint64_t>(bits_ops::ULessThan(lhs_bits, rhs_bits)),
              1));
        }
      }
      TernaryValue expected = ReduceFromBits(results)[0];
      TernaryValue actual = evaluator_.ULessThan(lhs, rhs);
      std::string message = absl::StrFormat("%s < %s => %s", ToString(lhs),
                                            ToString(rhs), ToString(expected));
      VLOG(1) << message;
      EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
    }
  }
}

TEST_F(TernaryLogicTest, BinarySelect) {
  for (const TernaryVector& selector : EnumerateTernaryVectors(/*width=*/1)) {
    for (const TernaryVector& on_true : EnumerateTernaryVectors(/*width=*/2)) {
      for (const TernaryVector& on_false :
           EnumerateTernaryVectors(/*width=*/2)) {
        std::vector<Bits> results;
        for (const Bits& selector_bits : ExpandToBits(selector)) {
          for (const Bits& on_true_bits : ExpandToBits(on_true)) {
            for (const Bits& on_false_bits : ExpandToBits(on_false)) {
              results.push_back(selector_bits.Get(0) ? on_true_bits
                                                     : on_false_bits);
            }
          }
        }
        TernaryVector expected = ReduceFromBits(results);
        TernaryVector actual = evaluator_.Select(selector, {on_false, on_true});
        std::string message = absl::StrFormat(
            "Sel(%s, cases=[%s, %s]) => %s", ToString(selector),
            ToString(on_false), ToString(on_true), ToString(expected));
        VLOG(1) << message;

        EXPECT_EQ(expected, actual)
            << message << ", but result is " << ToString(actual);
      }
    }
  }
}

TEST_F(TernaryLogicTest, ThreeWaySelectWithDefault) {
  for (const TernaryVector& selector : EnumerateTernaryVectors(/*width=*/3)) {
    for (const TernaryVector& case0 : EnumerateTernaryVectors(/*width=*/1)) {
      for (const TernaryVector& case1 : EnumerateTernaryVectors(/*width=*/1)) {
        for (const TernaryVector& case2 :
             EnumerateTernaryVectors(/*width=*/1)) {
          for (const TernaryVector& default_case :
               EnumerateTernaryVectors(/*width=*/1)) {
            std::vector<Bits> results;
            for (const Bits& selector_bits : ExpandToBits(selector)) {
              for (const Bits& case0_bits : ExpandToBits(case0)) {
                for (const Bits& case1_bits : ExpandToBits(case1)) {
                  for (const Bits& case2_bits : ExpandToBits(case2)) {
                    for (const Bits& default_case_bits :
                         ExpandToBits(default_case)) {
                      if (selector_bits == UBits(0, 3)) {
                        results.push_back(case0_bits);
                      } else if (selector_bits == UBits(1, 3)) {
                        results.push_back(case1_bits);
                      } else if (selector_bits == UBits(2, 3)) {
                        results.push_back(case2_bits);
                      } else {
                        results.push_back(default_case_bits);
                      }
                    }
                  }
                }
              }
            }
            TernaryVector expected = ReduceFromBits(results);
            TernaryVector actual = evaluator_.Select(
                selector, {case0, case1, case2}, default_case);
            std::string message = absl::StrFormat(
                "Sel(%s, cases=[%s, %s, %s], default=%s) => %s",
                ToString(selector), ToString(case0), ToString(case1),
                ToString(case2), ToString(default_case), ToString(expected));
            VLOG(1) << message;

            EXPECT_EQ(expected, actual)
                << message << ", but result is " << ToString(actual);
          }
        }
      }
    }
  }
}

TEST_F(TernaryLogicTest, OneHotSelectSelectorCanBeZero) {
  // Enumerate all ternary inputs for a 3-wide selector with single bit
  // cases.
  for (const TernaryVector& selector : EnumerateTernaryVectors(/*width=*/3)) {
    for (const TernaryVector& a : EnumerateTernaryVectors(/*width=*/1)) {
      for (const TernaryVector& b : EnumerateTernaryVectors(/*width=*/1)) {
        for (const TernaryVector& c : EnumerateTernaryVectors(/*width=*/1)) {
          std::vector<Bits> results;
          for (const Bits& selector_bits : ExpandToBits(selector)) {
            for (const Bits& a_bits : ExpandToBits(a)) {
              for (const Bits& b_bits : ExpandToBits(b)) {
                for (const Bits& c_bits : ExpandToBits(c)) {
                  // Achievement unlocked: octuply nested loop.
                  results.push_back(bits_ops::Or(
                      bits_ops::And(selector_bits.Slice(0, 1), a_bits),
                      bits_ops::Or(
                          bits_ops::And(selector_bits.Slice(1, 1), b_bits),
                          bits_ops::And(selector_bits.Slice(2, 1), c_bits))));
                }
              }
            }
          }
          TernaryVector expected = ReduceFromBits(results);
          TernaryVector actual =
              evaluator_.OneHotSelect(selector, {a, b, c},
                                      /*selector_can_be_zero=*/true);
          std::string message = absl::StrFormat(
              "OneHotSel(%s, cases=[%s, %s, %s]) => %s", ToString(selector),
              ToString(a), ToString(b), ToString(c), ToString(expected));
          VLOG(1) << message;

          EXPECT_EQ(expected, actual)
              << message << ", but result is " << ToString(actual);
        }
      }
    }
  }
}

TEST_F(TernaryLogicTest, OneHotSelectSelectorCannotBeZero) {
  // Enumerate all ternary inputs for a 3-wide selector with single bit cases.
  for (const TernaryVector& selector : EnumerateTernaryVectors(/*width=*/3)) {
    for (const TernaryVector& a : EnumerateTernaryVectors(/*width=*/1)) {
      for (const TernaryVector& b : EnumerateTernaryVectors(/*width=*/1)) {
        for (const TernaryVector& c : EnumerateTernaryVectors(/*width=*/1)) {
          std::vector<Bits> results;
          for (const Bits& selector_bits : ExpandToBits(selector)) {
            if (selector_bits.IsZero()) {
              continue;
            }
            for (const Bits& a_bits : ExpandToBits(a)) {
              for (const Bits& b_bits : ExpandToBits(b)) {
                for (const Bits& c_bits : ExpandToBits(c)) {
                  results.push_back(bits_ops::Or(
                      bits_ops::And(selector_bits.Slice(0, 1), a_bits),
                      bits_ops::Or(
                          bits_ops::And(selector_bits.Slice(1, 1), b_bits),
                          bits_ops::And(selector_bits.Slice(2, 1), c_bits))));
                }
              }
            }
          }
          if (results.empty()) {
            continue;
          }
          TernaryVector expected = ReduceFromBits(results);
          TernaryVector actual =
              evaluator_.OneHotSelect(selector, {a, b, c},
                                      /*selector_can_be_zero=*/false);
          std::string message = absl::StrFormat(
              "OneHotSel(%s, cases=[%s, %s, %s]) && selector != 0 => %s",
              ToString(selector), ToString(a), ToString(b), ToString(c),
              ToString(expected));
          VLOG(1) << message;

          EXPECT_EQ(expected, actual)
              << message << ", but result is " << ToString(actual);
        }
      }
    }
  }
}

TEST_F(TernaryLogicTest, ShiftRightLogical3wide) {
  // Enumerate all pairs of 3-wide ternary inputs.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/3)) {
    for (const TernaryVector& amount : EnumerateTernaryVectors(/*width=*/3)) {
      std::vector<Bits> results;
      for (const Bits& input_bits : ExpandToBits(input)) {
        for (const Bits& amount_bits : ExpandToBits(amount)) {
          results.push_back(bits_ops::ShiftRightLogical(
              input_bits, amount_bits.ToUint64().value()));
        }
      }
      TernaryVector expected = ReduceFromBits(results);
      TernaryVector actual = evaluator_.ShiftRightLogical(input, amount);
      std::string message =
          absl::StrFormat("%s >> %s => %s", ToString(input), ToString(amount),
                          ToString(expected));
      VLOG(1) << message;
      EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
    }
  }
}

TEST_F(TernaryLogicTest, ShiftRightLogical4x2) {
  // Enumerate all pairs of 4-wide inputs with a 2-wide shifter.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/4)) {
    for (const TernaryVector& amount : EnumerateTernaryVectors(/*width=*/2)) {
      std::vector<Bits> results;
      for (const Bits& input_bits : ExpandToBits(input)) {
        for (const Bits& amount_bits : ExpandToBits(amount)) {
          results.push_back(bits_ops::ShiftRightLogical(
              input_bits, amount_bits.ToUint64().value()));
        }
      }
      TernaryVector expected = ReduceFromBits(results);
      TernaryVector actual = evaluator_.ShiftRightLogical(input, amount);
      std::string message =
          absl::StrFormat("%s >> %s => %s", ToString(input), ToString(amount),
                          ToString(expected));
      VLOG(1) << message;
      EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
    }
  }
}

TEST_F(TernaryLogicTest, ShiftRightArith3wide) {
  // Enumerate all pairs of 3-wide ternary inputs.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/3)) {
    for (const TernaryVector& amount : EnumerateTernaryVectors(/*width=*/3)) {
      std::vector<Bits> results;
      for (const Bits& input_bits : ExpandToBits(input)) {
        for (const Bits& amount_bits : ExpandToBits(amount)) {
          results.push_back(bits_ops::ShiftRightArith(
              input_bits, amount_bits.ToUint64().value()));
        }
      }
      TernaryVector expected = ReduceFromBits(results);
      TernaryVector actual = evaluator_.ShiftRightArith(input, amount);
      std::string message =
          absl::StrFormat("%s >>> %s => %s", ToString(input), ToString(amount),
                          ToString(expected));
      VLOG(1) << message;
      EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
    }
  }
}

TEST_F(TernaryLogicTest, ShiftRightArith4x2) {
  // Enumerate all pairs of 4-wide inputs with a 2-wide shifter.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/4)) {
    for (const TernaryVector& amount : EnumerateTernaryVectors(/*width=*/2)) {
      std::vector<Bits> results;
      for (const Bits& input_bits : ExpandToBits(input)) {
        for (const Bits& amount_bits : ExpandToBits(amount)) {
          results.push_back(bits_ops::ShiftRightArith(
              input_bits, amount_bits.ToUint64().value()));
        }
      }
      TernaryVector expected = ReduceFromBits(results);
      TernaryVector actual = evaluator_.ShiftRightArith(input, amount);
      std::string message =
          absl::StrFormat("%s >>> %s => %s", ToString(input), ToString(amount),
                          ToString(expected));
      VLOG(1) << message;
      EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
    }
  }
}

TEST_F(TernaryLogicTest, ShiftLeftLogical3wide) {
  // Enumerate all pairs of 3-wide ternary inputs.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/3)) {
    for (const TernaryVector& amount : EnumerateTernaryVectors(/*width=*/3)) {
      std::vector<Bits> results;
      for (const Bits& input_bits : ExpandToBits(input)) {
        for (const Bits& amount_bits : ExpandToBits(amount)) {
          results.push_back(bits_ops::ShiftLeftLogical(
              input_bits, amount_bits.ToUint64().value()));
        }
      }
      TernaryVector expected = ReduceFromBits(results);
      TernaryVector actual = evaluator_.ShiftLeftLogical(input, amount);
      std::string message =
          absl::StrFormat("%s << %s => %s", ToString(input), ToString(amount),
                          ToString(expected));
      VLOG(1) << message;
      EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
    }
  }
}

TEST_F(TernaryLogicTest, ShiftLeftLogical4x2) {
  // Enumerate all pairs of 4-wide inputs with a 2-wide shifter.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/4)) {
    for (const TernaryVector& amount : EnumerateTernaryVectors(/*width=*/2)) {
      std::vector<Bits> results;
      for (const Bits& input_bits : ExpandToBits(input)) {
        for (const Bits& amount_bits : ExpandToBits(amount)) {
          results.push_back(bits_ops::ShiftLeftLogical(
              input_bits, amount_bits.ToUint64().value()));
        }
      }
      TernaryVector expected = ReduceFromBits(results);
      TernaryVector actual = evaluator_.ShiftLeftLogical(input, amount);
      std::string message =
          absl::StrFormat("%s << %s => %s", ToString(input), ToString(amount),
                          ToString(expected));
      VLOG(1) << message;
      EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
    }
  }
}

TEST_F(TernaryLogicTest, OneHotLsbToMsb) {
  // Enumerate all pairs of 4-wide ternary inputs.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/4)) {
    std::vector<Bits> results;
    for (const Bits& input_bits : ExpandToBits(input)) {
      results.push_back(bits_ops::OneHotLsbToMsb(input_bits));
    }
    TernaryVector expected = ReduceFromBits(results);
    TernaryVector actual = evaluator_.OneHotLsbToMsb(input);
    std::string message = absl::StrFormat("OneHotLsbToMsb(%s) => %s",
                                          ToString(input), ToString(expected));
    VLOG(1) << message;
    EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
  }
}

TEST_F(TernaryLogicTest, Decode) {
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b0"), 0)), "0b");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b1"), 0)), "0b");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0bX"), 0)), "0b");

  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b0"), 1)), "0b1");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b0"), 2)), "0b01");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b1"), 1)), "0b0");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b1"), 2)), "0b10");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0bX"), 1)), "0bX");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0bX"), 2)), "0bXX");

  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b0000"), 16)),
            "0b0000_0000_0000_0001");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b1001"), 16)),
            "0b0000_0010_0000_0000");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b1111"), 16)),
            "0b1000_0000_0000_0000");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0bXXXX"), 16)),
            "0bXXXX_XXXX_XXXX_XXXX");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0bXXX0"), 16)),
            "0b0X0X_0X0X_0X0X_0X0X");
  EXPECT_EQ(ToString(evaluator_.Decode(FromString("0b0XX1"), 16)),
            "0b0000_0000_X0X0_X0X0");
}

TEST_F(TernaryLogicTest, Encode) {
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0"))), "0b");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b1"))), "0b");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0bX"))), "0b");

  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b00"))), "0b0");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b01"))), "0b0");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0bXX"))), "0bX");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b10"))), "0b1");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b11"))), "0b1");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0bX1"))), "0bX");

  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0000_0000"))), "0b000");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0000_0001"))), "0b000");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0000_1000"))), "0b011");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b1000_0000"))), "0b111");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0001_0010"))), "0b101");

  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0000_000X"))), "0b000");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0000_00XX"))), "0b00X");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0b0000_0XXX"))), "0b0XX");
  EXPECT_EQ(ToString(evaluator_.Encode(FromString("0bXX00_010X"))), "0bX1X");
}

TEST_F(TernaryLogicTest, OneHotMsbToLsb) {
  // Enumerate all pairs of 4-wide ternary inputs.
  for (const TernaryVector& input : EnumerateTernaryVectors(/*width=*/4)) {
    std::vector<Bits> results;
    for (const Bits& input_bits : ExpandToBits(input)) {
      results.push_back(bits_ops::OneHotMsbToLsb(input_bits));
    }
    TernaryVector expected = ReduceFromBits(results);
    TernaryVector actual = evaluator_.OneHotMsbToLsb(input);
    std::string message = absl::StrFormat("OneHotMsbToLsb(%s) => %s",
                                          ToString(input), ToString(expected));
    VLOG(1) << message;
    EXPECT_EQ(expected, actual) << message << ", but result is " << actual;
  }
}

TEST_F(TernaryLogicTest, TestTheTestStuff) {
  EXPECT_THAT(EnumerateTernaryVectors(/*width=*/0),
              ElementsAre(FromString("0b")));
  EXPECT_THAT(
      EnumerateTernaryVectors(/*width=*/1),
      ElementsAre(FromString("0b0"), FromString("0b1"), FromString("0bx")));
  EXPECT_THAT(
      EnumerateTernaryVectors(/*width=*/2),
      ElementsAre(FromString("0b00"), FromString("0b10"), FromString("0bX0"),
                  FromString("0b01"), FromString("0b11"), FromString("0bX1"),
                  FromString("0b0X"), FromString("0b1X"), FromString("0bXX")));
  EXPECT_EQ(EnumerateTernaryVectors(/*width=*/5).size(), 243);

  EXPECT_THAT(ExpandToBits(FromString("0b")), ElementsAre(Bits()));
  EXPECT_THAT(ExpandToBits(FromString("0b1")), ElementsAre(UBits(1, 1)));
  EXPECT_THAT(ExpandToBits(FromString("0b0")), ElementsAre(UBits(0, 1)));
  EXPECT_THAT(ExpandToBits(FromString("0b01100111")),
              ElementsAre(UBits(0b01100111, 8)));

  EXPECT_THAT(ExpandToBits(FromString("0bX")),
              ElementsAre(UBits(0, 1), UBits(1, 1)));
  EXPECT_THAT(ExpandToBits(FromString("0bXX")),
              ElementsAre(UBits(0, 2), UBits(1, 2), UBits(2, 2), UBits(3, 2)));
  EXPECT_THAT(ExpandToBits(FromString("0b1X00X11")),
              ElementsAre(UBits(0b1000011, 7), UBits(0b1000111, 7),
                          UBits(0b1100011, 7), UBits(0b1100111, 7)));

  EXPECT_EQ(ReduceFromBits({UBits(0, 0)}), FromString("0b"));
  EXPECT_EQ(ReduceFromBits({UBits(0, 1), UBits(0, 1)}), FromString("0b0"));
  EXPECT_EQ(ReduceFromBits({UBits(1, 1), UBits(1, 1)}), FromString("0b1"));
  EXPECT_EQ(ReduceFromBits({UBits(0, 1), UBits(1, 1)}), FromString("0bX"));

  EXPECT_EQ(ReduceFromBits({UBits(0b1101, 4), UBits(0b0101, 4),
                            UBits(0b1100, 4), UBits(0b0100, 4)}),
            FromString("0bX10X"));
}

}  // namespace
}  // namespace xls
