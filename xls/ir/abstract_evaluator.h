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

#ifndef XLS_IR_ABSTRACT_EVALUATOR_H_
#define XLS_IR_ABSTRACT_EVALUATOR_H_

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"

namespace xls {

// An abstract base class for constructing evaluators which perform common XLS
// operations such as bitwise logical operations, arithmetic operations,
// selects, shifts, etc. "Bit" values in an evaluator can be arbitrarily typed
// enabling logical forms other than boolean logic. An example is ternary logic
// where the primitive type has three possible values: zero, one and unknown.
//
// The derived class provides the primitive type (as a template parameter) and a
// set of fundamental operations (OR, AND, and NOT) on the primitive types as
// well as the one and zero values. The AbstractEvaluator base class provides
// implementations of more complicated operations (for example, select or
// one-hot) as compositions of these fundamental operations.
//
// The AbstractEvaluator uses the curiously recurring template pattern to avoid
// the overhead of virtual functions. Classes should be derived like so:
//
//   class FooEvaluator : public AbstractEvaluator<FooValue, FooEvaluator> {
//     ...
//
// And define non-virtual methods One, Zero, Not, And, and Or.
template <typename ElementT, typename EvaluatorT>
class AbstractEvaluator {
 public:
  using Element = ElementT;
  using Vector = std::vector<Element>;

  Element One() const { return static_cast<const EvaluatorT*>(this)->One(); }
  Element Zero() const { return static_cast<const EvaluatorT*>(this)->Zero(); }
  Element Not(const Element& input) const {
    return static_cast<const EvaluatorT*>(this)->Not(input);
  }
  Element And(const Element& a, const Element& b) const {
    return static_cast<const EvaluatorT*>(this)->And(a, b);
  }
  Element Or(const Element& a, const Element& b) const {
    return static_cast<const EvaluatorT*>(this)->Or(a, b);
  }

  // Returns the given bits value as a Vector type.
  Vector BitsToVector(const Bits& bits) {
    Vector result(bits.bit_count());
    for (int64 i = 0; i < bits.bit_count(); ++i) {
      result[i] = bits.Get(i) ? One() : Zero();
    }
    return result;
  }

  Vector BitwiseNot(const Vector& input) {
    Vector result(input.size());
    for (int64 i = 0; i < input.size(); ++i) {
      result[i] = Not(input[i]);
    }
    return result;
  }

  // Bitwise n-ary logical operations.
  Vector BitwiseAnd(absl::Span<const Vector> inputs) {
    return NaryOp(
        inputs, [&](const Element& a, const Element& b) { return And(a, b); });
  }
  Vector BitwiseOr(absl::Span<const Vector> inputs) {
    return NaryOp(inputs,
                  [&](const Element& a, const Element& b) { return Or(a, b); });
  }
  Vector BitwiseXor(absl::Span<const Vector> inputs) {
    return NaryOp(inputs, [&](const Element& a, const Element& b) {
      return Or(And(Not(a), b), And(a, Not(b)));
    });
  }

  // Overloads of bitwise operations for two operands.
  Vector BitwiseAnd(const Vector& a, const Vector& b) {
    return BitwiseAnd({a, b});
  }
  Vector BitwiseOr(const Vector& a, const Vector& b) {
    return BitwiseOr({a, b});
  }
  Vector BitwiseXor(const Vector& a, const Vector& b) {
    return BitwiseXor({a, b});
  }

  Vector BitSlice(const Vector& input, int64 start, int64 width) {
    XLS_CHECK_GE(start, 0);
    XLS_CHECK_LE(start + width, input.size());
    XLS_CHECK_GE(width, 0);
    Vector result(width);
    for (int64 i = 0; i < width; ++i) {
      result[i] = input[start + i];
    }
    return result;
  }

  Vector Concat(absl::Span<const Vector> inputs) {
    Vector result;
    for (int64 i = inputs.size() - 1; i >= 0; --i) {
      result.insert(result.end(), inputs[i].begin(), inputs[i].end());
    }
    return result;
  }

  Element Equals(const Vector& a, const Vector& b) {
    XLS_CHECK_EQ(a.size(), b.size());
    Element result = One();
    for (int64 i = 0; i < a.size(); ++i) {
      result = And(result, Or(And(a[i], b[i]), And(Not(a[i]), Not(b[i]))));
    }
    return result;
  }

  Element SLessThan(const Vector& a, const Vector& b) {
    // A < B if:
    //  - A is negative && B is non-negative, or
    //  - B has a more-signficant bit set (when A and B have the same sign)
    // Which simplifies to:
    //  (A is neg & B is non-neg) ||
    //  (Not(A is non_neg & B is neg) & B has a more-significant bit set))
    // Which simplifies to:
    //  (A is neg & B is non-neg) ||
    //  (Not(A is non-neg and B is neg) && ULessThan(a, b))
    Element a_neg_b_non_neg = And(a.back(), Not(b.back()));
    Element a_non_neg_b_neg = And(Not(a.back()), b.back());
    return Or(a_neg_b_non_neg, And(Not(a_non_neg_b_neg), ULessThan(a, b)));
  }

  Element ULessThan(const Vector& a, const Vector& b) {
    XLS_CHECK_EQ(a.size(), b.size());
    Element result = Zero();
    Element upper_bits_lte = One();
    for (int64 i = a.size() - 1; i >= 0; --i) {
      result = Or(result, And(upper_bits_lte, And(Not(a[i]), b[i])));
      upper_bits_lte = And(upper_bits_lte, Or(Not(a[i]), b[i]));
    }
    return result;
  }

  Vector OneHotSelect(const Vector& selector, absl::Span<const Vector> cases,
                      bool selector_can_be_zero) {
    std::vector<absl::Span<const Element>> case_spans;
    case_spans.reserve(cases.size());
    for (const Vector& c : cases) {
      case_spans.push_back(c);
    }
    return OneHotSelectInternal(selector, case_spans, selector_can_be_zero);
  }

  Vector Select(const Vector& selector, absl::Span<const Vector> cases,
                absl::optional<const Vector> default_value = absl::nullopt) {
    // Turn the binary selector into a one-hot selector.
    Vector one_hot_selector;
    for (int64 i = 0; i < cases.size(); ++i) {
      one_hot_selector.push_back(
          Equals(selector, BitsToVector(UBits(i, selector.size()))));
    }
    // Copy the cases span as we may need to append to it.
    std::vector<Vector> cases_vec(cases.begin(), cases.end());
    if (default_value.has_value()) {
      one_hot_selector.push_back(ULessThan(
          BitsToVector(UBits(cases.size() - 1, selector.size())), selector));
      cases_vec.push_back(*default_value);
    }
    return OneHotSelect(one_hot_selector, cases_vec,
                        /*selector_can_be_zero=*/false);
  }

  // Performs an operation equivalent to the XLS IR Op::kOneHot
  // operation. OneHotMsbToLsb uses priority LsbOrMsb::kLsb, OneHotMsbToLsb uses
  // priority LsbOrMsb::kMsb
  Vector OneHotMsbToLsb(const Vector& input) {
    Element all_zero = One();
    Element any_one = Zero();
    Vector result;
    for (int64 i = input.size() - 1; i >= 0; --i) {
      result.push_back(And(all_zero, input[i]));
      all_zero = And(all_zero, Not(input[i]));
      any_one = Or(any_one, input[i]);
    }
    std::reverse(result.begin(), result.end());
    result.push_back(all_zero);
    return result;
  }
  Vector OneHotLsbToMsb(const Vector& input) {
    Element all_zero = One();
    Element any_one = Zero();
    Vector result;
    for (int64 i = 0; i < input.size(); ++i) {
      result.push_back(And(all_zero, input[i]));
      all_zero = And(all_zero, Not(input[i]));
      any_one = Or(any_one, input[i]);
    }
    result.push_back(all_zero);
    return result;
  }

  // Shifting by more than the width of the input results in all zeros for
  // logical shifts, and all sign bit for arithmetic shifts. Input and amount
  // need not be the same width.
  Vector ShiftRightLogical(const Vector& input, const Vector& amount) {
    return Shift(input, amount, /*right=*/true, /*arithmetic=*/false);
  }
  Vector ShiftRightArith(const Vector& input, const Vector& amount) {
    return Shift(input, amount, /*right=*/true, /*arithmetic=*/true);
  }
  Vector ShiftLeftLogical(const Vector& input, const Vector& amount) {
    return Shift(input, amount, /*right=*/false, /*arithmetic=*/false);
  }

  // Binary encode and decode operations.
  Vector Decode(const Vector& input, int64 result_width) {
    Vector result(result_width);
    for (int64 i = 0; i < result_width; ++i) {
      result[i] = Equals(input, BitsToVector(UBits(i, input.size())));
    }
    return result;
  }
  Vector Encode(const Vector& input) {
    int64 result_width = Bits::MinBitCountUnsigned(input.size() - 1);
    Vector result(result_width, Zero());
    for (int64 i = 0; i < input.size(); ++i) {
      for (int64 j = 0; j < result_width; ++j) {
        if ((i >> j) & 1) {
          result[j] = Or(result[j], input[i]);
        }
      }
    }
    return result;
  }

  Vector ZeroExtend(const Vector& input, int64 new_width) {
    XLS_CHECK_GE(new_width, input.size());
    return Concat({Vector(new_width - input.size(), Zero()), input});
  }

  Vector SignExtend(const Vector& input, int64 new_width) {
    XLS_CHECK_GE(input.size(), 1);
    XLS_CHECK_GE(new_width, input.size());
    return Concat({Vector(new_width - input.size(), input.back()), input});
  }

  // Reduction ops.
  Vector AndReduce(const Vector& a) {
    Element result = One();
    for (const Element& e : a) {
      result = And(result, e);
    }
    return Vector({result});
  }

  Vector OrReduce(const Vector& a) {
    Element result = Zero();
    for (const Element& e : a) {
      result = Or(result, e);
    }
    return Vector({result});
  }

  Vector XorReduce(const Vector& a) {
    Element result = Zero();
    for (const Element& e : a) {
      result = Or(And(Not(result), e), And(result, Not(e)));
    }
    return Vector({result});
  }

  Vector Add(const Vector& a, const Vector& b) {
    Vector result(a.size());
    Element carry = Zero();
    for (int i = 0; i < a.size(); i++) {
      Element a_i = a[i];
      Element b_i = b[i];
      Element not_a_i = Not(a[i]);
      Element not_b_i = Not(b[i]);
      Element not_carry = Not(carry);

      Vector bit_cases({
          And(not_a_i, And(not_b_i, carry)),
          And(not_a_i, And(b_i, not_carry)),
          And(a_i, And(not_b_i, not_carry)),
          And(a_i, And(b_i, carry)),
      });
      result[i] = OrReduce(bit_cases)[0];

      Vector carry_cases({
          And(not_a_i, And(b_i, carry)),
          And(a_i, And(not_b_i, carry)),
          And(a_i, And(b_i, not_carry)),
          And(a_i, And(b_i, carry)),
      });
      carry = OrReduce(carry_cases)[0];
    }
    return result;
  }

  Vector Neg(const Vector& x) {
    return Add(BitwiseNot(x), BitsToVector(UBits(1, x.size())));
  }

  // Signed multiplication of two Vectors.
  // Returns a Vector of a.size() + b.size().
  // Note: This is an inefficient implementation, but is adequate for immediate
  // needs. Algorithm is listed as "No Thinking Method" on
  // http://pages.cs.wisc.edu/~david/courses/cs354/beyond354/int.mult.html
  // (by Karen Miller).
  // This will be optimized in the future.
  Vector SMul(const Vector& a, const Vector& b) {
    int max = std::max(a.size(), b.size());
    Vector temp_a = SignExtend(a, max * 2);
    Vector temp_b = SignExtend(b, max * 2);
    Vector result = UMul(temp_a, temp_b);
    return BitSlice(result, 0, a.size() + b.size());
  }

  // Unsigned multiplication of two Vectors.
  // Returns a Vector of a.size() + b.size().
  Vector UMul(const Vector& a, const Vector& b) {
    Vector result(a.size() + b.size(), Zero());
    Element one = One();
    Vector zero_vector(a.size() + b.size(), Zero());
    Vector temp_b = ZeroExtend(b, result.size());
    for (const Element& e_a : a) {
      Vector addend = Select({e_a}, {zero_vector, temp_b});
      result = Add(result, addend);
      temp_b = ShiftLeftLogical(temp_b, {one});
    }
    return BitSlice(result, 0, a.size() + b.size());
  }

 private:
  // An implementation of OneHotSelect which takes a span of spans of Elements
  // rather than a span of Vectors. This enables the cases to be overlapping
  // spans of the same underlying vector as is used in the shift implementation.
  Vector OneHotSelectInternal(absl::Span<const Element> selector,
                              absl::Span<const absl::Span<const Element>> cases,
                              bool selector_can_be_zero) {
    XLS_CHECK_EQ(selector.size(), cases.size());
    XLS_CHECK_GT(selector.size(), 0);
    int64 width = cases.front().size();
    Vector result(width, Zero());
    for (int64 i = 0; i < selector.size(); ++i) {
      for (int64 j = 0; j < width; ++j) {
        result[j] = Or(result[j], And(cases[i][j], selector[i]));
      }
    }
    if (!selector_can_be_zero) {
      // If the selector cannot be zero, then a bit of the output can only be
      // zero if one of the respective bits of one of the cases is zero.
      // Construct such a mask and or it with the result.
      Vector and_reduction(width, One());
      for (int64 i = 0; i < selector.size(); ++i) {
        if (selector[i] != Zero()) {
          for (int64 j = 0; j < width; ++j) {
            and_reduction[j] = And(and_reduction[j], cases[i][j]);
          }
        }
      }
      result = BitwiseOr(and_reduction, result);
    }
    return result;
  }

  // Performs an N-ary logical operation on the given inputs. The operation is
  // defined by the given function.
  Vector NaryOp(absl::Span<const Vector> inputs,
                absl::FunctionRef<Element(const Element&, const Element&)> f) {
    XLS_CHECK_GT(inputs.size(), 0);
    Vector result(inputs.front());
    for (int64 i = 1; i < inputs.size(); ++i) {
      for (int64 j = 0; j < result.size(); ++j) {
        result[j] = f(result[j], inputs[i][j]);
      }
    }
    return result;
  }

  // Returns the result of shifting 'input' by' amount. If 'right' is true, then
  // a right-shift is performed. If 'arithmetic' is true, the shift is
  // arithmetic otherwise it is logical.
  Vector Shift(const Vector& input, const Vector& amount, bool right,
               bool arithmetic) {
    // Create the shift using a OneHotSelect. Each case of the OneHotSelect is
    // 'input' shifted by some constant value. The selector bits are each a
    // comparison of 'amount' to a constant value.
    //
    // First create a selector with input.size() + 1 bits containing the
    // following values:
    //   i in [0 ... input.size() - 1] : selector[i] = (amount == i)
    //   i == input.size()             : selector[i] = amount >= input.size()
    Vector selector;
    selector.reserve(input.size() + 1);
    auto bits_vector = [&](int64 v) {
      return BitsToVector(UBits(v, amount.size()));
    };
    for (int64 i = 0; i < input.size(); ++i) {
      if (amount.size() < Bits::MinBitCountUnsigned(i)) {
        // 'amount' doesn't have enough bits to express this shift amount.
        break;
      }
      selector.push_back(Equals(amount, bits_vector(i)));
    }
    // If 'amount' is wide enough to express over-shifting (shifting greater
    // than or equal to the input size), we need an additional case to catch
    // these instances.
    if (amount.size() >= Bits::MinBitCountUnsigned(input.size())) {
      selector.push_back(Not(ULessThan(amount, bits_vector(input.size()))));
    }

    // Create a span for each case in the one-hot-select. Each span corresponds
    // to the input vector shifted by a particular amount. Because of this
    // special structure of the spans, they may be represented a spans (slices)
    // of the same underlying vector. This is much faster than creating a
    // separate vector (and allocation) for each case.
    //
    // First create an extended version of the input vector where additional
    // bits are added to the beginning or end of the the vector. These
    // additional bits ensure that any shift amount corresponds to a particular
    // slice of the extended vector.
    //
    // In the comments below, the input vector is assumed to have the value
    // 'abcd' where 'a' through 'd' are the various bit values and 'd' is at
    // index 0.
    Vector extended;
    extended.reserve(input.size() + selector.size() - 1);
    std::vector<absl::Span<const Element>> cases;
    cases.reserve(selector.size());
    if (right) {
      // Shifting right.
      //
      // The 'extended' vector and the corresponding slices forming the various
      // cases are below.
      //
      // Arithmetic shift right ('a' is the sign bit):
      //
      //                 index
      //                7......0
      //  extended   =  aaaaabcd
      //    cases[0] =      abcd  // shra abcd, 0
      //    cases[1] =     aabc   // shra abcd, 1
      //    cases[2] =    aaab    //  ...
      //    cases[3] =   aaaa
      //    cases[4] =  aaaa
      //
      // Logical shift right:
      //
      //  extended   = 0000abcd
      //    cases[0] =     abcd   // shrl abcd, 0
      //    cases[1] =    0abc    // shrl abcd, 1
      //    cases[2] =   00ab     //  ...
      //    cases[3] =  000a
      //    cases[4] = 0000
      extended.insert(extended.begin(), input.begin(), input.end());
      for (int64 i = 0; i < selector.size() - 1; ++i) {
        extended.push_back(arithmetic ? input.back() : Zero());
      }
      for (int64 i = 0; i < selector.size(); ++i) {
        cases.push_back(absl::MakeConstSpan(&extended[i], input.size()));
      }
    } else {
      // Logical shift left:
      //
      //                 index
      //               7......0
      //  extended   = abcd0000
      //    cases[0] = abcd       // shll abcd, 0
      //    cases[1] =  bcd0      // shll abcd, 1
      //    cases[2] =   cd00     // ...
      //    cases[3] =    c000
      //    cases[4] =     0000
      for (int64 i = 0; i < selector.size() - 1; ++i) {
        extended.push_back(Zero());
      }
      for (int64 i = 0; i < input.size(); ++i) {
        extended.push_back(input[i]);
      }
      for (int64 i = 0; i < selector.size(); ++i) {
        cases.push_back(absl::MakeConstSpan(&extended[selector.size() - 1 - i],
                                            input.size()));
      }
    }
    return OneHotSelectInternal(selector, cases,
                                /*selector_can_be_zero=*/false);
  }
};

}  // namespace xls

#endif  // XLS_IR_ABSTRACT_EVALUATOR_H_
