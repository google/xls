// Copyright 2022 The XLS Authors
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

#ifndef XLS_FIXED_H
#define XLS_FIXED_H

#include <algorithm>

#include "xls_int.h"

#define __AC_NAMESPACE ac_datatypes
#include "external/com_github_hlslibs_ac_types/include/ac_fixed.h"
#include "external/com_github_hlslibs_ac_types/include/ac_int.h"
#ifndef __SYNTHESIS__
static_assert(false, "This header is only for synthesis");
#endif  // __SYNTHESIS__

// Quantize takes a value "in" that has been shifted, along with the original
// value then based on the quantization mode, it will either round the value or
// not. If it does round, it will do so by incrementing the value and setting
// the carry flag appropriately.
template <int W, int I, bool S, int FromW, int FromI, bool FromSigned,
          ac_datatypes::ac_q_mode q_mode, ac_datatypes::ac_o_mode o_mode>
class Quantize {
 public:
  inline static XlsInt<FromW, false> Adjust(__xls_bits<FromW> in,
                                            XlsInt<FromW, FromSigned> orig_val,
                                            bool &carry_out) {
    constexpr int F = W - I;
    constexpr int F2 = FromW - FromI;
    auto res = XlsInt<FromW, false>(in);
    if constexpr (q_mode != ac_datatypes::AC_TRN &&
                  !(q_mode == ac_datatypes::AC_TRN_ZERO && !FromSigned)) {
      bool qb;  // quantized bit
      if constexpr (F2 - F > FromW) {
        qb = (orig_val < 0);
      } else {
        qb = (bool)orig_val[F2 - F - 1];
      }
      bool rounded;
      if constexpr (F2 > F + 1) {
        XlsInt<F2 - F - 1, FromSigned> deleted_bits(0);
#pragma hls_unroll yes
        for (int i = 0; i < F2 - F - 1; ++i) {
          deleted_bits[i] = orig_val[i];
        }
        auto zero = XlsInt<F2 - F - 1>(0);
        rounded = (zero != deleted_bits);
      } else {
        rounded = false;
      }
      bool s = FromSigned && orig_val < 0;
      if constexpr (q_mode == ac_datatypes::AC_RND_ZERO) {
        qb &= s || rounded;
      } else if constexpr (q_mode == ac_datatypes::AC_RND_MIN_INF) {
        qb &= rounded;
      } else if constexpr (q_mode == ac_datatypes::AC_RND_INF) {
        qb &= !s || rounded;
      } else if constexpr (q_mode == ac_datatypes::AC_RND_CONV) {
        qb &= (res & 1) || rounded;
      } else if constexpr (q_mode == ac_datatypes::AC_RND_CONV_ODD) {
        qb &= (!(res & 1)) || rounded;
      } else if constexpr (q_mode == ac_datatypes::AC_TRN_ZERO) {
        qb = s && (qb || rounded);
      }
      carry_out = false;
      if (qb) {
        res++;
        carry_out = (res == 0);
      }
    }
    return res;
  }
};

// Overflow takes a set of bits, a boolean indicating if an overflow occurred
// while resizing to that value, and a boolean indicating if the input was
// negative. Based on the overflow mode, it will the appropriate value.
// These values can be found in section 2.1 of
// https://github.com/hlslibs/ac_types/blob/master/pdfdocs/ac_datatypes_ref.pdf
template <int W, bool S, ac_datatypes::ac_o_mode o_mode>
class Overflow {
 public:
  inline static __xls_bits<W> Adjust(__xls_bits<W> new_val, bool overflow,
                                     bool is_negative) {
    if constexpr (o_mode == ac_datatypes::AC_WRAP) {
      return new_val;
    } else if constexpr (o_mode == ac_datatypes::AC_SAT_ZERO) {
      if (overflow) {
        return XlsInt<W, S>(0).storage;
      } else {
        return new_val;
      }
    } else if constexpr (S) {
      if (overflow) {
        if (!is_negative) {
          return synth_only_internal::MaxValue<W, S>::Value();
        } else {
          if constexpr (o_mode == ac_datatypes::AC_SAT_SYM) {
            auto one = XlsInt<1, false>(1);
            return (XlsInt<W, S>(synth_only_internal::MinValue<W, S>::Value()) |
                    one)
                .storage;
          } else {
            return synth_only_internal::MinValue<W, S>::Value();
          }
        }
      } else {
        return new_val;
      }
    } else {
      if (overflow) {
        if (!is_negative) {
          return synth_only_internal::MaxValue<W, S>::Value();
        } else {
          return XlsInt<W, S>(0).storage;
        }
      } else {
        return new_val;
      }
    }
  }
};

// ResizeAndQuantize takes the original value "in" shifts it to the correct
// decimal position, then quantizes it if necessary, and resizes it to the
// correct bit width.
template <int W, int I, bool S, int FromW, int FromI, bool FromSigned,
          ac_datatypes::ac_q_mode q_mode = ac_datatypes::AC_TRN,
          ac_datatypes::ac_o_mode o_mode = ac_datatypes::AC_WRAP>
class ResizeAndQuantize {
 public:
  inline static __xls_bits<W> Adjust(__xls_bits<FromW> in, bool &carry_out) {
    constexpr int F = W - I;
    constexpr int F2 = FromW - FromI;
    const XlsInt<FromW, FromSigned> input_val(in);
    constexpr int shift =
        std::max((W - I) - (FromW - FromI), (FromW - FromI) - (W - I));
    constexpr unsigned int shift_log = Log2Ceil<shift> + 1;
    if constexpr (F2 == F) {
      if constexpr (W == FromW) {
        return in;
      } else if constexpr (W > FromW) {
        return ExtendBits<FromW, W, FromSigned>::Convert(in);
      } else {
        return SliceBits<FromW, W>::Convert(in);
      }
    } else if constexpr (F2 > F) {  // handle quantization
      auto offset = XlsInt<shift_log, false>(shift);
      auto shifted = ShiftRightWithSign<FromW, FromSigned, shift_log>::Operate(
          in, offset.storage);
      if constexpr (W == FromW) {
        auto res =
            Quantize<W, I, S, FromW, FromI, FromSigned, q_mode, o_mode>::Adjust(
                shifted, input_val, carry_out);
        return res.storage;
      } else if constexpr (W > FromW) {
        auto res =
            Quantize<W, I, S, FromW, FromI, FromSigned, q_mode, o_mode>::Adjust(
                shifted, input_val, carry_out);
        return ExtendBits<FromW, W, FromSigned>::Convert(res.storage);
      } else {
        auto res =
            Quantize<W, I, S, FromW, FromI, FromSigned, q_mode, o_mode>::Adjust(
                shifted, input_val, carry_out);
        return SliceBits<FromW, W>::Convert(res.storage);
      }
    } else {  // no quantization needed
      auto offset = XlsInt<shift_log, false>(shift);
      if constexpr (W == FromW) {
        return ShiftLeft<W, shift_log>::Operate(in, offset.storage);
      } else if constexpr (W > FromW) {
        auto extended = ExtendBits<FromW, W, FromSigned>::Convert(in);
        return ShiftLeft<W, shift_log>::Operate(extended, offset.storage);
      } else {
        auto shifted = ShiftLeft<FromW, shift_log>::Operate(in, offset.storage);
        return SliceBits<FromW, W>::Convert(shifted);
      }
    }
  }
};

// Adjustment resizes and if necessary quantizes the input using the supplied
// quantization mode and then applies overflow handling using the supplied
// overflow mode.
template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,
          ac_datatypes::ac_o_mode O, int W2, int I2, bool S2,
          ac_datatypes::ac_q_mode Q2, ac_datatypes::ac_o_mode O2>
class Adjustment {
 public:
  inline static __xls_bits<W> Adjust(__xls_bits<W2> in) {
    constexpr int F = W - I;
    constexpr int F2 = W2 - I2;
    constexpr bool QUAN_INC =
        F2 > F &&
        !(Q == ac_datatypes::AC_TRN || (Q == ac_datatypes::AC_TRN_ZERO && !S2));
    bool carry_out;
    __xls_bits<W> current_val =
        ResizeAndQuantize<W, I, S, W2, I2, S2, Q, O>::Adjust(in, carry_out);
    if constexpr (O != ac_datatypes::AC_WRAP &&
                  ((!S && S2) ||
                   I - S <
                       I2 - S2 +
                           (QUAN_INC ||
                            (S2 && O == ac_datatypes::AC_SAT_SYM &&
                             (O2 == ac_datatypes::AC_SAT_SYM || F2 > F))))) {
      const XlsInt<W2, S2> input_val(in);
      const XlsInt<W, S> current_val_int(current_val);

      bool deleted_bits_zero = true;
      bool deleted_bits_one = true;
      bool all_ones = true;
      if constexpr (I2 > I) {
#pragma hls_unroll yes
        for (int i = W2 - (I2 - I); i < W2; ++i) {
          deleted_bits_zero &= !input_val[i];
          deleted_bits_one &= input_val[i];
          all_ones &= input_val[i];
        }
        if (carry_out) {
          deleted_bits_zero = !deleted_bits_zero;
          deleted_bits_one = !deleted_bits_one;
        }
      }
      bool neg_src = S2 && input_val < 0 && !(carry_out & all_ones);
      bool neg_trg = S && current_val_int < 0;
      bool overflow = !neg_src && (neg_trg || !deleted_bits_zero);
      overflow |= neg_src && (!neg_trg || !deleted_bits_one);
      if constexpr (O == ac_datatypes::AC_SAT_SYM && S && S2) {
        overflow |= neg_src && (W > 1 ? deleted_bits_zero : true);
      }
      return Overflow<W, S, O>::Adjust(current_val, overflow, neg_src);
    } else {
      return current_val;
    }
  }
};

template <int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization = ac_datatypes::AC_TRN,
          ac_datatypes::ac_o_mode Overflow = ac_datatypes::AC_WRAP>
class [[hls_no_tuple]] XlsFixed {
 public:
  // XLS[cc] will initialize to 0
  inline XlsFixed() {}

  template <int FromW, bool FromSign>
  inline XlsFixed(const XlsInt<FromW, FromSign> &o)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow,
                       FromW, FromW, FromSign, ac_datatypes::AC_TRN,
                       ac_datatypes::AC_WRAP>::Adjust(o.storage))) {}

  inline XlsFixed(bool value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 1,
                       1, false, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<bool, 1>::Convert(value)))) {}

  inline XlsFixed(char value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 8,
                       8, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<char, 8>::Convert(value)))) {}

  inline XlsFixed(unsigned char value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 8,
                       8, false, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<unsigned char, 8>::Convert(value)))) {}

  inline XlsFixed(int value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 32,
                       32, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<int, 32>::Convert(value)))) {}

  inline XlsFixed(unsigned int value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 32,
                       32, false, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<unsigned int, 32>::Convert(value)))) {}

  inline XlsFixed(long value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 64,
                       64, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<long, 64>::Convert(value)))) {}

  inline XlsFixed(unsigned long value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 64,
                       64, false, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<unsigned long, 64>::Convert(value)))) {}

  inline XlsFixed(long long value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 64,
                       64, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<long long, 64>::Convert(value)))) {}

  inline XlsFixed(unsigned long long value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 64,
                       64, false, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(BuiltinIntToBits<unsigned long long, 64>::Convert(
                    value)))) {}

  // Undefined behavior if the double is out of 32 bit signed integer range
  inline XlsFixed(double value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 64,
                       32, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(__xlscc_fixed_32_32_bits_for_double(value)))) {}

  // Undefined behavior if the float is out of 32 bit signed integer range
  inline XlsFixed(float value)
      : val(XlsInt<Width, Signed>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, 64,
                       32, true, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP>::
                Adjust(__xlscc_fixed_32_32_bits_for_float(value)))) {}

  XlsInt<Width, false> val;
  static constexpr int width = Width;
  static constexpr int i_width = IntegerWidth;
  static constexpr bool sign = Signed;
  static constexpr ac_datatypes::ac_o_mode o_mode = Overflow;
  static constexpr ac_datatypes::ac_q_mode q_mode = Quantization;
  // Width+1 because we need to be able to represent shifting all digits
  typedef XlsInt<Log2Ceil<Width + 1>, false> index_t;

  inline bool is_neg() const { return sign && ((*this) < 0); }

  inline int to_int() const {
    auto ret(Adjustment<32, 32, true, ac_datatypes::AC_TRN,
                        ac_datatypes::AC_WRAP, Width, IntegerWidth, Signed,
                        Quantization, Overflow>::Adjust(val.storage));
    int reti;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_5_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(reti)
        : "i"(32), "a"(ret));
    return reti;
  }

  inline unsigned int to_uint() const {
    auto ret(Adjustment<32, 32, false, ac_datatypes::AC_TRN,
                        ac_datatypes::AC_WRAP, Width, IntegerWidth, Signed,
                        Quantization, Overflow>::Adjust(val.storage));
    return BitsToBuiltinInt<unsigned int, 32>::Convert(ret);
  }

  inline long to_long() const {
    auto ret(Adjustment<64, 64, true, ac_datatypes::AC_TRN,
                        ac_datatypes::AC_WRAP, Width, IntegerWidth, Signed,
                        Quantization, Overflow>::Adjust(val.storage));
    return BitsToBuiltinInt<long, 64>::Convert(ret);
  }

  inline unsigned long to_ulong() const {
    auto ret(Adjustment<64, 64, false, ac_datatypes::AC_TRN,
                        ac_datatypes::AC_WRAP, Width, IntegerWidth, Signed,
                        Quantization, Overflow>::Adjust(val.storage));
    return BitsToBuiltinInt<unsigned long, 64>::Convert(ret);
  }

  inline long long to_int64() const {
    auto ret(Adjustment<64, 64, true, ac_datatypes::AC_TRN,
                        ac_datatypes::AC_WRAP, Width, IntegerWidth, Signed,
                        Quantization, Overflow>::Adjust(val.storage));
    return BitsToBuiltinInt<long long, 64>::Convert(ret);
  }

  inline unsigned long long to_uint64() const {
    XlsFixed<64, 64, false, ac_datatypes::AC_TRN, ac_datatypes::AC_WRAP> ret(
        *this);
    return BitsToBuiltinInt<unsigned long long, 64>::Convert(ret.val.storage);
  }

  inline XlsInt<std::max(IntegerWidth, 1), Signed> to_ac_int() const {
    return ((XlsFixed<std::max(IntegerWidth, 1), std::max(IntegerWidth, 1),
                      Signed>)*this)
        .template slc<std::max(IntegerWidth, 1)>(0);
  }

  template <int W2, int I2, bool S2, ac_datatypes::ac_q_mode Q2,
            ac_datatypes::ac_o_mode O2>
  inline XlsFixed(const XlsFixed<W2, I2, S2, Q2, O2> &op)
      : val(XlsInt<Width, false>(
            Adjustment<Width, IntegerWidth, Signed, Quantization, Overflow, W2,
                       I2, S2, Q2, O2>::Adjust(op.val.storage))) {}

  // Defines the result types for each operation based on ac_int
  template <int ToW, int ToI, bool ToSign>
  struct rt
      : public ac_datatypes::ac_fixed<Width, IntegerWidth,
                                      Signed>::template rt<ToW, ToI, ToSign> {
    typedef XlsFixed<rt::mult_w, rt::mult_i, rt::mult_s> mult;
    typedef XlsFixed<rt::plus_w, rt::plus_i, rt::plus_s> plus;
    typedef XlsFixed<rt::minus_w, rt::minus_i, rt::minus_s> minus;
    typedef XlsFixed<rt::logic_w, rt::logic_i, rt::logic_s> logic;
    typedef XlsFixed<rt::div_w, rt::div_i, rt::div_s> div;
    typedef XlsFixed arg1;
    typedef XlsFixed ident;
  };

  struct rt_unary
      : public ac_datatypes::ac_fixed<Width, IntegerWidth, Signed, Quantization,
                                      Overflow>::rt_unary {
    typedef XlsFixed<rt_unary::neg_w, rt_unary::neg_i, rt_unary::neg_s,
                     Quantization, Overflow>
        neg;
    typedef XlsFixed<Width + !Signed, IntegerWidth + !Signed, true> bnot;
  };

  bool operator!() const { return (*this) == XlsFixed(0); }

  inline typename rt_unary::neg operator-() const {
    typename rt_unary::neg as = *this;
    typename rt_unary::neg ret;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret (aid): bits[i] "
        "= neg(a, pos=(loc)) }"
        : "=r"(ret)
        : "i"(rt_unary::neg::width), "parama"(as.val.storage));
    return ret;
  }

  // Sign extends
  inline typename rt_unary::bnot operator~() const {
    typename rt_unary::bnot as = *this;
    typename rt_unary::bnot ret;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret (aid): bits[i] "
        "= not(a, pos=(loc)) }"
        : "=r"(ret)
        : "i"(rt_unary::bnot::width), "parama"(as.val.storage));
    return ret;
  }

  // Doesn't sign extend
  inline XlsFixed<Width, IntegerWidth, false> bit_complement() const {
    XlsFixed<Width, IntegerWidth, false> as = *this;
    XlsFixed<Width, IntegerWidth, false> ret;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret (aid): bits[i] "
        "= not(a, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "parama"(as.val.storage));
    return ret;
  }

  inline XlsFixed operator+() const { return *this; }

  inline XlsFixed operator++() {
    val = val + 1;
    return *this;
  }

  inline XlsFixed operator++(int) {
    const XlsFixed orig = (*this);
    val = val + 1;
    return orig;
  }

  inline XlsFixed operator--() {
    val = val - 1;
    return *this;
  }

  inline XlsFixed operator--(int) {
    const XlsFixed orig = (*this);
    val = val - 1;
    return orig;
  }

#define BINARY_OP_FIXED(__OP, __IR, __RES)                                  \
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,     \
            ac_datatypes::ac_o_mode ToO>                                    \
  inline typename rt<ToW, ToI, ToSign>::__RES operator __OP(                \
      const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) const {                \
    typedef typename rt<ToW, ToI, ToSign>::__RES Result;                    \
    Result ret;                                                             \
    auto adj_a =                                                            \
        XlsFixed<Result::width, Result::i_width, Result::sign>(*this);      \
    auto adj_b = XlsFixed<Result::width, Result::i_width, Result::sign>(o); \
    auto a = adj_a.val.storage;                                             \
    auto b = adj_b.val.storage;                                             \
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[i] { ret (aid): bits[i] " \
        "= " __IR "(a, b, pos=(loc)) }"                                     \
        : "=r"(ret)                                                         \
        : "i"(Result::width), "parama"(a), "paramb"(b));                    \
    return ret;                                                             \
  }                                                                         \
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,     \
            ac_datatypes::ac_o_mode ToO>                                    \
  inline XlsFixed operator __OP##=(                                         \
      const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) {                      \
    (*this) = (*this)__OP o;                                                \
    return (*this);                                                         \
  }

  BINARY_OP_FIXED(+, "add", plus);
  BINARY_OP_FIXED(-, "sub", minus);

#define BINARY_LOGIC_OP_FIXED(__OP, __IR, __RES)                               \
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,        \
            ac_datatypes::ac_o_mode ToO>                                       \
  inline typename rt<ToW, ToI, ToSign>::__RES operator __OP(                   \
      const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) const {                   \
    typedef typename rt<ToW, ToI, ToSign>::__RES Result;                       \
    Result ret;                                                                \
    constexpr int F = Width - IntegerWidth;                                    \
    constexpr int F2 = ToW - ToI;                                              \
    if constexpr (F == F2) {                                                   \
      auto a = ExtendBits<Width, Result::width, Signed>::Convert(              \
          this->val.storage);                                                  \
      auto b = ExtendBits<ToW, Result::width, ToSign>::Convert(o.val.storage); \
      asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[i] { ret (aid): bits[i] "  \
          "= " __IR "(a, b, pos=(loc)) }"                                      \
          : "=r"(ret)                                                          \
          : "i"(Result::width), "parama"(a), "paramb"(b));                     \
    } else if constexpr (F > F2) {                                             \
      auto a = ExtendBits<Width, Result::width, Signed>::Convert(              \
          this->val.storage);                                                  \
      auto b1 =                                                                \
          ExtendBits<ToW, Result::width, ToSign>::Convert(o.val.storage);      \
      XlsInt<10, false> F_minus_F2(F - F2);                                    \
      auto b = ShiftLeft<Result::width, 10>::Operate(b1, F_minus_F2.storage);  \
      asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[i] { ret (aid): bits[i] "  \
          "= " __IR "(a, b, pos=(loc)) }"                                      \
          : "=r"(ret)                                                          \
          : "i"(Result::width), "parama"(a), "paramb"(b));                     \
    } else {                                                                   \
      auto a1 = ExtendBits<Width, Result::width, Signed>::Convert(             \
          this->val.storage);                                                  \
      XlsInt<10, false> F2_minus_F(F2 - F);                                    \
      auto a = ShiftLeft<Result::width, 10>::Operate(a1, F2_minus_F.storage);  \
      auto b = ExtendBits<ToW, Result::width, ToSign>::Convert(o.val.storage); \
      asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[i] { ret (aid): bits[i] "  \
          "= " __IR "(a, b, pos=(loc)) }"                                      \
          : "=r"(ret)                                                          \
          : "i"(Result::width), "parama"(a), "paramb"(b));                     \
    }                                                                          \
    return ret;                                                                \
  }                                                                            \
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,        \
            ac_datatypes::ac_o_mode ToO>                                       \
  inline XlsFixed operator __OP##=(                                            \
      const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) {                         \
    (*this) = (*this)__OP o;                                                   \
    return (*this);                                                            \
  }

  BINARY_LOGIC_OP_FIXED(|, "or", logic);
  BINARY_LOGIC_OP_FIXED(&, "and", logic);
  BINARY_LOGIC_OP_FIXED(^, "xor", logic);

  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,
            ac_datatypes::ac_o_mode ToO>
  inline typename rt<ToW, ToI, ToSign>::div operator/(
      const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) const {
    typedef typename rt<ToW, ToI, ToSign>::div Result;
    Result ret;
    constexpr int Num_w = Width + AC_MAX(ToW - ToI, 0);
    constexpr int Num_i = IntegerWidth;
    XlsFixed<Num_w, Num_i, Signed> a(*this);
    auto adj_a =
        ConvertBits<Num_w, Result::width, Signed>::Convert(a.val.storage);
    auto adj_b =
        ConvertBits<ToW, Result::width, ToSign>::Convert(o.val.storage);
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_4_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret.val.storage)
        : "i"(Result::width),
          "parama"(DivideWithSign<Result::width, Result::sign>::Operate(
              adj_a, adj_b)));
    return ret;
  }
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,
            ac_datatypes::ac_o_mode ToO>
  inline XlsFixed operator/=(const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) {
    (*this) = (*this) / o;
    return (*this);
  }

  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,
            ac_datatypes::ac_o_mode ToO>
  inline typename rt<ToW, ToI, ToSign>::mult operator*(
      const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) const {
    typedef typename rt<ToW, ToI, ToSign>::mult Result;
    Result ret;
    auto a =
        ConvertBits<Width, Result::width, Signed>::Convert(this->val.storage);
    auto b = ConvertBits<ToW, Result::width, ToSign>::Convert(o.val.storage);
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_4_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret.val.storage)
        : "i"(Result::width),
          "parama"(
              MultiplyWithSign<Result::width, Result::sign>::Operate(a, b)));
    return ret;
  }
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode ToQ,
            ac_datatypes::ac_o_mode ToO>
  inline XlsFixed operator*=(const XlsFixed<ToW, ToI, ToSign, ToQ, ToO> &o) {
    (*this) = (*this) * o;
    return (*this);
  }

  template <int W2, int I2, bool S2, ac_datatypes::ac_q_mode Q2,
            ac_datatypes::ac_o_mode O2>
  inline XlsFixed operator>>(XlsFixed<W2, I2, S2, Q2, O2> offset) const {
    XlsFixed<W2, I2, S2> neg_offset = -offset;
    XlsFixed ret_right;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_5_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret_right.val.storage)
        : "i"(Width), "a"(ShiftRightWithSign<Width, Signed, W2>::Operate(
                          this->val.storage, offset.val.storage)));
    XlsFixed ret_left;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shll(a, o, pos=(loc)) }"
        : "=r"(ret_left.val.storage)
        : "i"(Width), "c"(W2), "a"(this->val.storage),
          "o"(neg_offset.val.storage));
    XlsFixed<32, 32, S2> offset_trunc = offset;  // to match ac_fixed use 32 bit
    return (offset_trunc < 0) ? ret_left : ret_right;
  }
  template <int W2, int I2, bool S2, ac_datatypes::ac_q_mode Q2,
            ac_datatypes::ac_o_mode O2>
  inline XlsFixed operator>>=(XlsFixed<W2, I2, S2, Q2, O2> offset) {
    (*this) = (*this) >> offset;
    return (*this);
  }

  template <int W2, int I2, bool S2, ac_datatypes::ac_q_mode Q2,
            ac_datatypes::ac_o_mode O2>
  inline XlsFixed operator<<(XlsFixed<W2, I2, S2, Q2, O2> offset) const {
    XlsFixed<W2, I2, S2> neg_offset = -offset;
    XlsFixed ret_right;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_5_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret_right.val.storage)
        : "i"(Width), "a"(ShiftRightWithSign<Width, Signed, W2>::Operate(
                          this->val.storage, neg_offset.val.storage)));
    XlsFixed ret_left;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shll(a, o, pos=(loc)) }"
        : "=r"(ret_left.val.storage)
        : "i"(Width), "c"(W2), "a"(this->val.storage), "o"(offset.val.storage));
    XlsFixed<32, 32, S2> offset_trunc = offset;  // to match ac_fixed
    return (offset_trunc < 0) ? ret_right : ret_left;
  }

  template <int W2, int I2, bool S2, ac_datatypes::ac_q_mode Q2,
            ac_datatypes::ac_o_mode O2>
  inline XlsFixed operator<<=(XlsFixed<W2, I2, S2, Q2, O2> offset) {
    (*this) = (*this) << offset;
    return (*this);
  }

#define COMPARISON_OP_FIXED(__OP, __IR)                                     \
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode Q2,      \
            ac_datatypes::ac_o_mode O2>                                     \
  inline bool operator __OP(const XlsFixed<ToW, ToI, ToSign, Q2, O2> &o)    \
      const {                                                               \
    typedef typename rt<ToW, ToI, ToSign>::logic Result;                    \
    Result fixed(o);                                                        \
    Result a(*this);                                                        \
    bool ret;                                                               \
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[1] { ret (aid): bits[1] " \
        "= " __IR "(a, b, pos=(loc)) }"                                     \
        : "=r"(ret)                                                         \
        : "i"(Result::width), "parama"(a.val.storage),                      \
          "paramb"(fixed.val.storage));                                     \
    return ret;                                                             \
  }

  COMPARISON_OP_FIXED(==, "eq");

  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode Q2,
            ac_datatypes::ac_o_mode O2>
  inline bool operator!=(const XlsFixed<ToW, ToI, ToSign, Q2, O2> &o) const {
    return !((*this) == o);
  }

#define COMPARISON_OP_FIXED_WITH_SIGN(__OP, __IMPL)                      \
  template <int ToW, int ToI, bool ToSign, ac_datatypes::ac_q_mode Q2,   \
            ac_datatypes::ac_o_mode O2>                                  \
  inline bool operator __OP(const XlsFixed<ToW, ToI, ToSign, Q2, O2> &o) \
      const {                                                            \
    typedef typename rt<ToW, ToI, ToSign>::logic Result;                 \
    Result fixed(o);                                                     \
    Result a(*this);                                                     \
    bool ret;                                                            \
    asm("fn (fid)(a: bits[i]) -> bits[1] { ret op_6_(aid): bits[1] = "   \
        "identity(a, pos=(loc)) }"                                       \
        : "=r"(ret)                                                      \
        : "i"(1), "parama"(__IMPL<Result::width, Result::sign>::Operate( \
                      a.val.storage, fixed.val.storage)));               \
    return ret;                                                          \
  }

  COMPARISON_OP_FIXED_WITH_SIGN(>, GreaterWithSign);
  COMPARISON_OP_FIXED_WITH_SIGN(>=, GreaterOrEqWithSign);
  COMPARISON_OP_FIXED_WITH_SIGN(<, LessWithSign);
  COMPARISON_OP_FIXED_WITH_SIGN(<=, LessOrEqWithSign);

  inline XlsFixed operator=(const XlsFixed &o) {
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_7_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(this->val.storage)
        : "i"(Width), "a"(o));
    return *this;
  }

  template <int ToW>
  inline XlsInt<ToW, Signed> slc(index_t offset) const {
    static_assert(index_t::width > 0, "Bug in Log2Ceil");
    static_assert(Width >= ToW, "Can't take a slice wider than the source");
    static_assert(Width > 0, "Can't slice 0 bits");
    static_assert(ToW > 0, "Can't slice 0 bits");
    typedef XlsInt<ToW, Signed> Result;
    Result ret;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[r] { ret op_(aid): bits[r] = "
        "dynamic_bit_slice(a, o, width=r, pos=(loc)) }"
        : "=r"(ret.storage)
        : "i"(Width), "r"(ToW), "c"(index_t::width), "a"(this->val.storage),
          "o"(offset.storage));
    return ret;
  }

  inline bool operator[](index_t i) const {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[1] { ret op_(aid): bits[1] = "
        "dynamic_bit_slice(a, o, width=1, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "c"(index_t::width), "a"(this->val.storage),
          "o"(i.storage));
    return ret;
  }

  // --- Hack: see comments for BitElemRef
  inline BitElemRef operator[](index_t i) {
    return BitElemRef((bool)slc<1>(i));

    // NOP to ensure that clang parses the set_element functions
    set_element_bitref(0, 1);
    set_element_xlsint(0, 1);
    set_element_int(0, 1);
  }

  inline XlsFixed set_element_bitref(index_t i, BitElemRef rvalue) {
    set_slc(i, XlsInt<1, false>(rvalue));
    return *this;
  }

  inline XlsFixed set_element_int(index_t i, int rvalue) {
    set_slc(i, XlsInt<1, false>(rvalue));
    return *this;
  }

  inline XlsFixed set_element_xlsint(index_t i, XlsInt<1, false> rvalue) {
    set_slc(i, rvalue);
    return *this;
  }
  // --- / Hack

  inline index_t clz() const {
    XlsFixed<Width, IntegerWidth, false, Quantization, Overflow> reverse_out;
    asm("fn (fid)(a: bits[i]) -> bits[i] { "
        "  ret (aid): bits[i] = reverse(a, pos=(loc)) }"
        : "=r"(reverse_out.val.storage)
        : "i"(Width), "a"(this->val.storage));

    XlsFixed<Width + 1, IntegerWidth, false, Quantization, Overflow>
        one_hot_out;
    asm("fn (fid)(a: bits[i]) -> bits[c] { "
        "  ret (aid): bits[c] = one_hot(a, lsb_prio=true, pos=(loc)) }"
        : "=r"(one_hot_out.val.storage)
        : "i"(Width), "c"(Width + 1), "a"(reverse_out));

    index_t encode_out;
    asm("fn (fid)(a: bits[i]) -> bits[c] { "
        "  ret (aid): bits[c] = encode(a, pos=(loc)) }"
        : "=r"(encode_out.storage)
        : "i"(Width + 1), "c"(index_t::width), "a"(one_hot_out));

    // zero_ext is automatic
    return encode_out;
  }

  // Counts leading bits. For unsigned values, it's clz,
  // for signed, the sign bit is ignored, and leading 1s are counted
  // for negative values.
  inline index_t leading_sign() const {
    if (Signed) {
      int ret = 0;
      if ((*this) < 0) {
        ret = (~(*this)).clz();
      } else {
        ret = clz();
      }
      return ret - 1;
    }
    return clz();
  }

  inline index_t leading_sign(bool &all_sign) const {
    index_t ls = leading_sign();
    all_sign = (ls == Width - Signed);
    return ls;
  }

  template <int ToW, bool ToSign>
  XlsFixed set_slc(index_t offset, XlsInt<ToW, ToSign> slice_raw) {
    static_assert(index_t::width > 0, "Bug in Log2Ceil");
    static_assert(Width >= ToW, "Can't set a slice wider than the source");
    static_assert(Width > 0, "Can't slice 0 bits");
    static_assert(ToW > 0, "Can't slice 0 bits");
    asm("fn (fid)(a: bits[i], o: bits[c], u: bits[d]) -> bits[i] { ret "
        "op_(aid): bits[i] = "
        "bit_slice_update(a, o, u, pos=(loc)) }"
        : "=r"(this->val.storage)
        : "i"(Width), "c"(index_t::width), "d"(ToW), "a"(this->val.storage),
          "o"(offset.storage), "u"(slice_raw.storage));
    return *this;
  }

  template <int W2, bool S2>
  inline XlsFixed operator>>(XlsInt<W2, S2> offset) const {
    XlsInt<W2, S2> neg_offset = -offset;
    XlsFixed ret_right;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_5_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret_right.val.storage)
        : "i"(Width), "a"(ShiftRightWithSign<Width, Signed, W2>::Operate(
                          this->val.storage, offset.storage)));
    XlsFixed ret_left;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shll(a, o, pos=(loc)) }"
        : "=r"(ret_left.val.storage)
        : "i"(Width), "c"(W2), "a"(this->val.storage), "o"(neg_offset.storage));
    return (offset < 0) ? ret_left : ret_right;
  }

  template <int W2, bool S2>
  inline XlsFixed operator>>=(XlsInt<W2, S2> offset) {
    (*this) = (*this) >> offset;
    return (*this);
  }

  template <int W2, bool S2>
  inline XlsFixed operator<<(XlsInt<W2, S2> offset) const {
    XlsInt<W2, true> neg_offset = -offset;
    XlsFixed ret_right;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_5_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret_right.val.storage)
        : "i"(Width), "a"(ShiftRightWithSign<Width, Signed, W2>::Operate(
                          this->val.storage, neg_offset.storage)));
    XlsFixed ret_left;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shll(a, o, pos=(loc)) }"
        : "=r"(ret_left.val.storage)
        : "i"(Width), "c"(W2), "a"(this->val.storage), "o"(offset.storage));
    return (offset < 0) ? ret_right : ret_left;
  }

  template <int W2, bool S2>
  inline XlsFixed operator<<=(XlsInt<W2, S2> offset) {
    (*this) = (*this) << offset;
    return (*this);
  }

  template <int ToW, int ToI, bool ToSign,
            ac_datatypes::ac_q_mode ToQuantization,
            ac_datatypes::ac_o_mode ToOverflow>
  friend class XlsFixed;
};

template <int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
inline std::ostream &operator<<(
    std::ostream &os,
    const XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow> &x) {
  return os;
}

#define ac_fixed XlsFixed

// XLS[cc] doesn't support returning a reference
#undef FX_ASSIGN_OP_WITH_INT_2
#define FX_ASSIGN_OP_WITH_INT_2(ASSIGN_OP, C_TYPE, W2, S2)                    \
  template <int W, int I, bool S>                                             \
  inline XlsFixed<W, I, S> operator ASSIGN_OP(XlsInt<W, S> &op, C_TYPE op2) { \
    return op.operator ASSIGN_OP(XlsFixed<W2, W2, S2>(op2));                  \
  }

#undef FX_ASSIGN_OP_WITH_INT_2I
#define FX_ASSIGN_OP_WITH_INT_2I(ASSIGN_OP, C_TYPE, W2, S2)          \
  template <int W, int I, bool S>                                    \
  inline XlsFixed<W, I, S> operator ASSIGN_OP(XlsFixed<W, I, S> &op, \
                                              C_TYPE op2) {          \
    return op.operator ASSIGN_OP(XlsInt<W2, S2>(op2));               \
  }

#undef FX_BIN_OP_WITH_INT_2I
#define FX_BIN_OP_WITH_INT_2I(BIN_OP, C_TYPE, WI, SI)        \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q, \
            ac_datatypes::ac_o_mode O>                       \
  inline XlsFixed<W, I, S, Q, O> operator BIN_OP(            \
      const XlsFixed<W, I, S, Q, O> &op, C_TYPE i_op) {      \
    return op.operator BIN_OP(XlsInt<WI, SI>(i_op));         \
  }

#undef FX_BIN_OP_WITH_INT
#define FX_BIN_OP_WITH_INT(BIN_OP, C_TYPE, WI, SI, RTYPE)           \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,        \
            ac_datatypes::ac_o_mode O>                              \
  inline typename XlsFixed<WI, WI, SI>::template rt<W, I, S>::RTYPE \
  operator BIN_OP(C_TYPE i_op, const XlsFixed<W, I, S, Q, O> &op) { \
    return XlsFixed<WI, WI, SI>(i_op).operator BIN_OP(op);          \
  }                                                                 \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,        \
            ac_datatypes::ac_o_mode O>                              \
  inline typename XlsFixed<W, I, S>::template rt<WI, WI, SI>::RTYPE \
  operator BIN_OP(const XlsFixed<W, I, S, Q, O> &op, C_TYPE i_op) { \
    return op.operator BIN_OP(XlsFixed<WI, WI, SI>(i_op));          \
  }

#undef FX_REL_OP_WITH_INT
#define FX_REL_OP_WITH_INT(REL_OP, C_TYPE, W2, S2)                             \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,                   \
            ac_datatypes::ac_o_mode O>                                         \
  inline bool operator REL_OP(const XlsFixed<W, I, S, Q, O> &op, C_TYPE op2) { \
    return op.operator REL_OP(XlsFixed<W2, W2, S2>(op2));                      \
  }                                                                            \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,                   \
            ac_datatypes::ac_o_mode O>                                         \
  inline bool operator REL_OP(C_TYPE op2, const XlsFixed<W, I, S, Q, O> &op) { \
    return XlsFixed<W2, W2, S2>(op2).operator REL_OP(op);                      \
  }

FX_OPS_WITH_INT(bool, 1, false)
FX_OPS_WITH_INT(signed char, 8, true)
FX_OPS_WITH_INT(unsigned char, 8, false)
FX_OPS_WITH_INT(short, 16, true)
FX_OPS_WITH_INT(unsigned short, 16, false)
FX_OPS_WITH_INT(int, 32, true)
FX_OPS_WITH_INT(unsigned int, 32, false)
FX_OPS_WITH_INT(long, 64, true)
FX_OPS_WITH_INT(unsigned long, 64, false)
FX_OPS_WITH_INT(long long, 64, true)
FX_OPS_WITH_INT(unsigned long long, 64, false)

#undef FX_BIN_OP_WITH_AC_INT_1
#define FX_BIN_OP_WITH_AC_INT_1(BIN_OP, RTYPE)                      \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,        \
            ac_datatypes::ac_o_mode O, int WI, bool SI>             \
  inline typename XlsFixed<WI, WI, SI>::template rt<W, I, S>::RTYPE \
  operator BIN_OP(const XlsInt<WI, SI> &i_op,                       \
                  const XlsFixed<W, I, S, Q, O> &op) {              \
    return XlsFixed<WI, WI, SI>(i_op).operator BIN_OP(op);          \
  }

#undef FX_BIN_OP_WITH_AC_INT_2
#define FX_BIN_OP_WITH_AC_INT_2(BIN_OP, RTYPE)                      \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,        \
            ac_datatypes::ac_o_mode O, int WI, bool SI>             \
  inline typename XlsFixed<W, I, S>::template rt<WI, WI, SI>::RTYPE \
  operator BIN_OP(const XlsFixed<W, I, S, Q, O> &op,                \
                  const XlsInt<WI, SI> &i_op) {                     \
    return op.operator BIN_OP(XlsFixed<WI, WI, SI>(i_op));          \
  }

FX_BIN_OP_WITH_AC_INT(*, mult)
FX_BIN_OP_WITH_AC_INT(+, plus)
FX_BIN_OP_WITH_AC_INT(-, minus)
FX_BIN_OP_WITH_AC_INT(/, div)
FX_BIN_OP_WITH_AC_INT(&, logic)
FX_BIN_OP_WITH_AC_INT(|, logic)
FX_BIN_OP_WITH_AC_INT(^, logic)

#undef FX_REL_OP_WITH_AC_INT
#define FX_REL_OP_WITH_AC_INT(REL_OP)                              \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,       \
            ac_datatypes::ac_o_mode O, int WI, bool SI>            \
  inline bool operator REL_OP(const XlsFixed<W, I, S, Q, O> &op,   \
                              const XlsInt<WI, SI> &op2) {         \
    return op.operator REL_OP(XlsFixed<WI, WI, SI>(op2));          \
  }                                                                \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,       \
            ac_datatypes::ac_o_mode O, int WI, bool SI>            \
  inline bool operator REL_OP(XlsInt<WI, SI> &op2,                 \
                              const XlsFixed<W, I, S, Q, O> &op) { \
    return XlsFixed<WI, WI, SI>(op2).operator REL_OP(op);          \
  }

FX_REL_OP_WITH_AC_INT(==)
FX_REL_OP_WITH_AC_INT(!=)
FX_REL_OP_WITH_AC_INT(>)
FX_REL_OP_WITH_AC_INT(>=)
FX_REL_OP_WITH_AC_INT(<)
FX_REL_OP_WITH_AC_INT(<=)

#undef XLS_FX_ASSIGN_OP_WITH_AC_INT
#define XLS_FX_ASSIGN_OP_WITH_AC_INT(ASSIGN_OP)                 \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,    \
            ac_datatypes::ac_o_mode O, int WI, bool SI>         \
  inline const XlsFixed<W, I, S, Q, O> operator ASSIGN_OP(      \
      XlsFixed<W, I, S, Q, O> &op, const XlsInt<WI, SI> &op2) { \
    return op.operator ASSIGN_OP(XlsFixed<WI, WI, SI>(op2));    \
  }                                                             \
  template <int W, int I, bool S, ac_datatypes::ac_q_mode Q,    \
            ac_datatypes::ac_o_mode O, int WI, bool SI>         \
  inline const XlsInt<WI, SI> operator ASSIGN_OP(               \
      XlsInt<WI, SI> &op, const XlsFixed<W, I, S, Q, O> &op2) { \
    return op.operator ASSIGN_OP(op2.to_ac_int());              \
  }

XLS_FX_ASSIGN_OP_WITH_AC_INT(+=)
XLS_FX_ASSIGN_OP_WITH_AC_INT(-=)
XLS_FX_ASSIGN_OP_WITH_AC_INT(*=)
XLS_FX_ASSIGN_OP_WITH_AC_INT(/=)
XLS_FX_ASSIGN_OP_WITH_AC_INT(&=)
XLS_FX_ASSIGN_OP_WITH_AC_INT(|=)
XLS_FX_ASSIGN_OP_WITH_AC_INT(^=)

#undef ac_fixed

#endif  // XLS_FIXED_H
