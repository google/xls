// Copyright 2021 The XLS Authors
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

#ifndef XLS_INT_H
#define XLS_INT_H

#include <algorithm>
#include <type_traits>

#include "/xls_builtin.h"

#define __AC_NAMESPACE ac_datatypes
#include "include/ac_int.h"

#ifndef __SYNTHESIS__
static_assert(false, "This header is only for synthesis");
#endif  // __SYNTHESIS__

namespace {

template <int rem>
static const int Log2Floor = (rem <= 1) ? 0 : (1 + Log2Floor<rem / 2>);
template <>
const int Log2Floor<0> = 0;
template <int n>
static const int Log2Ceil = Log2Floor<n> + (((1 << Log2Floor<n>) == n) ? 0 : 1);
template <>
const int Log2Ceil<1> = 1;

static_assert(Log2Ceil<1> == 1);
static_assert(Log2Ceil<2> == 1);
static_assert(Log2Ceil<3> == 2);
static_assert(Log2Ceil<4> == 2);
static_assert(Log2Ceil<5> == 3);
static_assert(Log2Ceil<15> == 4);
static_assert(Log2Ceil<16> == 4);
static_assert(Log2Ceil<17> == 5);

template <typename T, int Width>
class BuiltinIntToBits {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Convert(T in) {
    __xls_bits<Width> in_bits;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_9_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(in_bits)
        : "i"(Width), "a"(in));
    return in_bits;
  }
};

#pragma hls_synthetic_int
template <typename T, int Width>
class BitsToBuiltinInt {
 public:
  inline static T Convert(__xls_bits<Width> in) {
    T in_type;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_9_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(in_type)
        : "i"(Width), "a"(in));
    return in_type;
  }
};

template <int FromW, int ToW, bool Signed>
class ExtendBits {};

template <int FromW, int ToW>
class ExtendBits<FromW, ToW, false> {
 public:
  static_assert(FromW < ToW);

#pragma hls_synthetic_int
  inline static __xls_bits<ToW> Convert(__xls_bits<FromW> storage) {
    __xls_bits<ToW> ret;
    asm("fn (fid)(a: bits[i]) -> bits[d] { ret (aid): bits[d] = "
        "zero_ext(a, new_bit_count=d, pos=(loc)) }"
        : "=r"(ret)
        : "i"(FromW), "d"(ToW), "parama"(storage));
    return ret;
  }
};

template <int FromW, int ToW>
class ExtendBits<FromW, ToW, true> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<ToW> Convert(__xls_bits<FromW> storage) {
    __xls_bits<ToW> ret;
    asm("fn (fid)(a: bits[i]) -> bits[d] { ret (aid): bits[d] = "
        "sign_ext(a, new_bit_count=d, pos=(loc)) }"
        : "=r"(ret)
        : "i"(FromW), "d"(ToW), "parama"(storage));
    return ret;
  }
};

template <int FromW, int ToW>
class SliceBits {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<ToW> Convert(__xls_bits<FromW> storage) {
    __xls_bits<ToW> ret;
    asm("fn (fid)(a: bits[i]) -> bits[d] { ret (aid): bits[d] = "
        "bit_slice(a, start=0, width=d, pos=(loc)) }"
        : "=r"(ret)
        : "i"(FromW), "d"(ToW), "parama"(storage));
    return ret;
  }
};

template <int Width>
class PassThroughBits {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Convert(__xls_bits<Width> storage) {
    return storage;
  }
};

template <int FromW, int ToW, bool Signed>
class ConvertBits
    : public std::conditional<(ToW == FromW), PassThroughBits<FromW>,
                              typename std::conditional<
                                  (ToW > FromW), ExtendBits<FromW, ToW, Signed>,
                                  SliceBits<FromW, ToW> >::type>::type {};

template <int Width, bool Signed>
class MultiplyWithSign {};

template <int Width>
class MultiplyWithSign<Width, false> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<Width> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[d] { ret (aid): bits[d] "
        "= umul(a, b, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "d"(Width), "parama"(a), "paramb"(b));
    return ret;
  }
};

template <int Width>
class MultiplyWithSign<Width, true> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<Width> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[d] { ret (aid): bits[d] "
        "= smul(a, b, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "d"(Width), "parama"(a), "paramb"(b));
    return ret;
  }
};

#pragma hls_synthetic_int
template <int Width, int IndexW>
class ShiftLeft {
 public:
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<IndexW> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shll(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "c"(IndexW), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, bool Signed, int IndexW>
class ShiftRightWithSign {};

template <int Width, int IndexW>
class ShiftRightWithSign<Width, false, IndexW> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<IndexW> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shrl(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "c"(IndexW), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, int IndexW>
class ShiftRightWithSign<Width, true, IndexW> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<IndexW> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shra(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "c"(IndexW), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, bool Signed>
class DivideWithSign {};

template <int Width>
class DivideWithSign<Width, false> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<Width> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[i] { ret op_(aid): bits[i] = "
        "udiv(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width>
class DivideWithSign<Width, true> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<Width> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[i] { ret op_(aid): bits[i] = "
        "sdiv(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, bool Signed>
class ModuloWithSign {};

template <int Width>
class ModuloWithSign<Width, false> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<Width> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[i] { ret op_(aid): bits[i] = "
        "umod(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width>
class ModuloWithSign<Width, true> {
 public:
#pragma hls_synthetic_int
  inline static __xls_bits<Width> Operate(__xls_bits<Width> a,
                                          __xls_bits<Width> b) {
    __xls_bits<Width> ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[i] { ret op_(aid): bits[i] = "
        "smod(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, bool Signed>
class GreaterWithSign {};

template <int Width>
class GreaterWithSign<Width, false> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "ugt(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width>
class GreaterWithSign<Width, true> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "sgt(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, bool Signed>
class GreaterOrEqWithSign {};

template <int Width>
class GreaterOrEqWithSign<Width, false> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "uge(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width>
class GreaterOrEqWithSign<Width, true> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "sge(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, bool Signed>
class LessWithSign {};

template <int Width>
class LessWithSign<Width, false> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "ult(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width>
class LessWithSign<Width, true> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "slt(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width, bool Signed>
class LessOrEqWithSign {};

template <int Width>
class LessOrEqWithSign<Width, false> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "ule(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

template <int Width>
class LessOrEqWithSign<Width, true> {
 public:
#pragma hls_synthetic_int
  inline static bool Operate(__xls_bits<Width> a, __xls_bits<Width> b) {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[i]) -> bits[1] { ret op_(aid): bits[1] = "
        "sle(a, o, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "a"(a), "o"(b));
    return ret;
  }
};

}  // namespace

// The point of XlsIntBase is to provide different conversions
//  for signed and unsigned ints. It is the base class to XlsInt.
#pragma hls_synthetic_int
template <int Width, bool Signed>
class XlsIntBase {};

#pragma hls_synthetic_int
template <int Width>
class XlsIntBase<Width, false> {
  static_assert(Width > 0, "Must be at least 1 bit wide");

 public:
  // XLS[cc] will init with 0
  inline XlsIntBase() {}

  inline XlsIntBase(const __xls_bits<Width> o) : storage(o) {}

  inline operator unsigned long long() const {
    static_assert(Width <= 64);
    __xls_bits<64> ret(ConvertBits<Width, 64, false>::Convert(storage));
    unsigned long long reti;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_0_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(reti)
        : "i"(64), "a"(ret));
    return reti;
  }

  explicit inline operator bool() const {
    __xls_bits<Width> zero(ConvertBits<32, Width, false>::Convert(
            BuiltinIntToBits<unsigned int, 32>::Convert(0)));
    bool reti;
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[1] { ret op_0_(aid): bits[1] = "
        "ne(a, b, pos=(loc)) }"
        : "=r"(reti)
        : "i"(Width), "a"(storage), "b"(zero));
    return reti;
  }

  __xls_bits<Width> storage;
};

#pragma hls_synthetic_int
template <int Width>
class XlsIntBase<Width, true> {
  static_assert(Width > 0, "Must be at least 1 bit wide");

 public:
  // XLS[cc] will init with 0
  inline XlsIntBase() {}

  inline XlsIntBase(__xls_bits<Width> o) : storage(o) {}

  inline operator long long() const {
    static_assert(Width <= 64);
    __xls_bits<64> ret(ConvertBits<Width, 64, true>::Convert(storage));
    long long reti;

    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_1_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(reti)
        : "i"(64), "a"(ret));

    return reti;
  }

  explicit inline operator bool() const {
    __xls_bits<Width> zero(ConvertBits<32, Width, false>::Convert(
            BuiltinIntToBits<unsigned int, 32>::Convert(0)));
    bool reti;
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[1] { ret op_0_(aid): bits[1] = "
        "ne(a, b, pos=(loc)) }"
        : "=r"(reti)
        : "i"(Width), "a"(storage), "b"(zero));
    return reti;
  }

  __xls_bits<Width> storage;
};
template <int W, bool S>
class XlsInt;

// BitElemRef is returned from XlsInt's non-const operator [].
// It represents a reference to a certain bit inside an XlsInt.
//
// TODO(seanhaskell): This class does not contain a reference to an XlsInt
//   because XLS[cc] doesn't support returning references yet. Instead,
//   XLS[cc] contains a hack to explicitly handle this case.
#pragma hls_no_tuple
struct BitElemRef {
  // template <typename T>
  inline BitElemRef(bool in) : v(in) {}

  inline operator bool() const { return v; }
  template <int W2, bool S2>
  operator XlsInt<W2, S2>() const {
    return operator bool();
  }

  inline BitElemRef operator=(int val) {
    v = static_cast<bool>(val);
    return *this;
  }

  inline BitElemRef operator=(const BitElemRef &val) {
    v = static_cast<bool>(val.v);
    return *this;
  }

  bool v;
};

#pragma hls_synthetic_int
template <int Width, bool Signed = true>
class XlsInt : public XlsIntBase<Width, Signed> {
 public:
  // XLS[cc] will initialize to 0
  inline XlsInt() {}

  template <int ToW, bool ToSign>
  inline XlsInt(const XlsInt<ToW, ToSign> &o)
      : XlsIntBase<Width, Signed>(
            ConvertBits<ToW, Width, ToSign>::Convert(o.storage)) {}

  inline XlsInt(bool value)
      : XlsIntBase<Width, Signed>(ConvertBits<1, Width, false>::Convert(
            BuiltinIntToBits<bool, 1>::Convert(value))) {}

  inline XlsInt(char value)
      : XlsIntBase<Width, Signed>(ConvertBits<8, Width, true>::Convert(
            BuiltinIntToBits<char, 8>::Convert(value))) {}

  inline XlsInt(unsigned char value)
      : XlsIntBase<Width, Signed>(ConvertBits<8, Width, false>::Convert(
            BuiltinIntToBits<unsigned char, 8>::Convert(value))) {}

  inline XlsInt(int value)
      : XlsIntBase<Width, Signed>(ConvertBits<32, Width, true>::Convert(
            BuiltinIntToBits<int, 32>::Convert(value))) {}

  inline XlsInt(unsigned int value)
      : XlsIntBase<Width, Signed>(ConvertBits<32, Width, false>::Convert(
            BuiltinIntToBits<unsigned int, 32>::Convert(value))) {}

  inline XlsInt(long value)
      : XlsIntBase<Width, Signed>(ConvertBits<64, Width, true>::Convert(
            BuiltinIntToBits<long, 64>::Convert(value))) {}

  inline XlsInt(unsigned long value)
      : XlsIntBase<Width, Signed>(ConvertBits<64, Width, false>::Convert(
            BuiltinIntToBits<unsigned long, 64>::Convert(value))) {}

  inline XlsInt(long long value)
      : XlsIntBase<Width, Signed>(ConvertBits<64, Width, true>::Convert(
            BuiltinIntToBits<long long, 64>::Convert(value))) {}

  inline XlsInt(unsigned long long value)
      : XlsIntBase<Width, Signed>(ConvertBits<64, Width, false>::Convert(
            BuiltinIntToBits<unsigned long long, 64>::Convert(value))) {}

  inline XlsInt(const double value)
      : XlsIntBase<Width, Signed>(ConvertBits<64, Width, true>::Convert(
            BuiltinIntToBits<long long, 64>::Convert((long long)value))) {}

  inline XlsInt(const float value)
      : XlsIntBase<Width, Signed>(ConvertBits<64, Width, true>::Convert(
            BuiltinIntToBits<long long, 64>::Convert((long long)value))) {}

  inline XlsInt(__xls_bits<Width> value) : XlsIntBase<Width, Signed>(value) {}

  inline int to_int() const {
    XlsInt<32, true> ret(*this);

    int reti;

    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_2_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(reti)
        : "i"(32), "a"(ret));

    return reti;
  }

  inline unsigned int to_uint() const { return (unsigned int)to_int(); }
  inline long long to_int64() const {
    XlsInt<64, true> ret(*this);
    long long reti;

    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_1_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(reti)
        : "i"(64), "a"(ret));

    return reti;
  }
  inline unsigned long long to_uint64() const {
    return (unsigned long long)to_int64();
  }
  inline long to_long() const { return (long)to_int64(); }
  inline unsigned long to_ulong() const { return (unsigned long)to_int64(); }

  static const int width = Width;
  static const int i_width = Width;
  static const bool sign = Signed;

  // Defines the result types for each operation based on ac_int
  template <int ToW, bool ToSign>
  struct rt
      : public ac_datatypes::ac_int<Width, Signed>::template rt<ToW, ToSign> {
    typedef XlsInt<rt::mult_w, rt::mult_s> mult;
    typedef XlsInt<rt::plus_w, rt::plus_s> plus;
    typedef XlsInt<rt::minus_w, rt::minus_s> minus;
    typedef XlsInt<rt::logic_w, rt::logic_s> logic;
    typedef XlsInt<rt::div_w, rt::div_s> div;
    typedef XlsInt<rt::mod_w, rt::mod_s> mod;
    typedef XlsInt arg1;
    typedef XlsInt ident;
  };

  struct rt_unary : public ac_datatypes::ac_int<Width, Signed>::rt_unary {
    typedef XlsInt<rt_unary::neg_w, rt_unary::neg_s> neg;
    typedef XlsInt<Width + !Signed, true> bnot;
  };

  bool operator!() const { return (*this) == XlsInt(0); }

  XlsInt operator+() const { return (*this); }

  inline typename rt_unary::neg operator-() const {
    typename rt_unary::neg as = *this;
    typename rt_unary::neg ret;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret (aid): bits[i] "
        "= neg(a, pos=(loc)) }"
        : "=r"(ret)
        : "i"(rt_unary::neg::width), "parama"(as));
    return ret;
  }

  // Sign extends
  inline typename rt_unary::bnot operator~() const {
    typename rt_unary::bnot as = *this;
    typename rt_unary::bnot ret;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret (aid): bits[i] "
        "= not(a, pos=(loc)) }"
        : "=r"(ret)
        : "i"(rt_unary::bnot::width), "parama"(as));
    return ret;
  }

  // Doesn't sign extend
  inline XlsInt<Width, false> bit_complement() const {
    XlsInt<Width, false> as = *this;
    XlsInt<Width, false> ret;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret (aid): bits[i] "
        "= not(a, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "parama"(as));
    return ret;
  }

  inline XlsInt operator++() {
    (*this) = (*this) + 1;
    return (*this);
  }

  inline XlsInt operator++(int) {
    const XlsInt orig = (*this);
    (*this) = (*this) + 1;
    return orig;
  }

  inline XlsInt operator--() {
    (*this) = (*this) - 1;
    return (*this);
  }

  inline XlsInt operator--(int) {
    const XlsInt orig = (*this);
    (*this) = (*this) - 1;
    return orig;
  }

#define BINARY_OP(__OP, __IR, __RES)                                        \
  template <int ToW, bool ToSign>                                           \
  inline typename rt<ToW, ToSign>::__RES operator __OP(                     \
      const XlsInt<ToW, ToSign> &o) const {                                 \
    typedef typename rt<ToW, ToSign>::__RES Result;                         \
    Result as = *this;                                                      \
    Result bs = o;                                                          \
    Result ret;                                                             \
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[i] { ret (aid): bits[i] " \
        "= " __IR "(a, b, pos=(loc)) }"                                     \
        : "=r"(ret)                                                         \
        : "i"(Result::width), "parama"(as), "paramb"(bs));                  \
    return ret;                                                             \
  }                                                                         \
  template <int ToW, bool ToSign>                                           \
  inline XlsInt operator __OP##=(const XlsInt<ToW, ToSign> &o) {            \
    (*this) = (*this)__OP o;                                                \
    return (*this);                                                         \
  }

#define BINARY_OP_WITH_SIGN(__OP, __IMPL, __RES)                               \
  template <int ToW, bool ToSign>                                              \
  inline typename rt<ToW, ToSign>::__RES operator __OP(                        \
      const XlsInt<ToW, ToSign> &o) const {                                    \
    typedef typename rt<ToW, ToSign>::__RES Result;                            \
    Result as = *this;                                                         \
    Result bs = o;                                                             \
    Result ret;                                                                \
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_4_(aid): bits[i] = "         \
        "identity(a, pos=(loc)) }"                                             \
        : "=r"(ret.storage)                                                    \
        : "i"(Result::width),                                                  \
          "parama"(__IMPL<Result::width, Result::sign>::Operate(as.storage,    \
                                                                bs.storage))); \
    return ret;                                                                \
  }                                                                            \
  template <int ToW, bool ToSign>                                              \
  inline XlsInt operator __OP##=(const XlsInt<ToW, ToSign> &o) {               \
    (*this) = (*this)__OP o;                                                   \
    return (*this);                                                            \
  }

  BINARY_OP(+, "add", plus);
  BINARY_OP(-, "sub", minus);

  BINARY_OP_WITH_SIGN(*, MultiplyWithSign, mult);
  BINARY_OP_WITH_SIGN(/, DivideWithSign, div);
  BINARY_OP_WITH_SIGN(%, ModuloWithSign, mod);

  BINARY_OP(|, "or", logic);
  BINARY_OP(&, "and", logic);
  BINARY_OP(^, "xor", logic);

  template <int W2, bool S2>
  inline XlsInt operator>>(XlsInt<W2, S2> offset) const {
    XlsInt<W2, S2> neg_offset = -offset;
    XlsInt ret_right;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_5_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret_right.storage)
        : "i"(Width), "a"(ShiftRightWithSign<Width, Signed, W2>::Operate(
                          this->storage, offset.storage)));
    XlsInt ret_left;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shll(a, o, pos=(loc)) }"
        : "=r"(ret_left.storage)
        : "i"(Width), "c"(W2), "a"(this->storage), "o"(neg_offset.storage));
    return (offset < 0) ? ret_left : ret_right;
  }
  template <int W2, bool S2>
  inline XlsInt operator>>=(XlsInt<W2, S2> offset) {
    (*this) = (*this) >> offset;
    return (*this);
  }

  template <int W2, bool S2>
  inline XlsInt operator<<(XlsInt<W2, S2> offset) const {
    XlsInt<W2, S2> neg_offset = -offset;
    XlsInt ret_right;
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_5_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(ret_right.storage)
        : "i"(Width), "a"(ShiftRightWithSign<Width, Signed, W2>::Operate(
                          this->storage, neg_offset.storage)));
    XlsInt ret_left;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[i] { ret op_(aid): bits[i] = "
        "shll(a, o, pos=(loc)) }"
        : "=r"(ret_left.storage)
        : "i"(Width), "c"(W2), "a"(this->storage), "o"(offset.storage));
    return (offset < 0) ? ret_right : ret_left;
  }
  template <int W2, bool S2>
  inline XlsInt operator<<=(XlsInt<W2, S2> offset) {
    (*this) = (*this) << offset;
    return (*this);
  }

#define COMPARISON_OP(__OP, __IR)                                           \
  template <int ToW, bool ToSign>                                           \
  inline bool operator __OP(const XlsInt<ToW, ToSign> &o) const {           \
    XlsInt val(o);                                                          \
    bool ret;                                                               \
    asm("fn (fid)(a: bits[i], b: bits[i]) -> bits[1] { ret (aid): bits[1] " \
        "= " __IR "(a, b, pos=(loc)) }"                                     \
        : "=r"(ret)                                                         \
        : "i"(Width), "parama"(this->storage), "paramb"(val.storage));      \
    return ret;                                                             \
  }

  COMPARISON_OP(==, "eq");

  template <int ToW, bool ToSign>
  inline bool operator!=(const XlsInt<ToW, ToSign> &o) const {
    return !((*this) == o);
  }

#define COMPARISON_OP_WITH_SIGN(__OP, __IMPL)                             \
  template <int ToW, bool ToSign>                                         \
  inline bool operator __OP(const XlsInt<ToW, ToSign> &o) const {         \
    XlsInt<std::max(ToW, Width), Signed | ToSign> valA(o);                \
    XlsInt<std::max(ToW, Width), Signed | ToSign> valB(*this);            \
    bool ret;                                                             \
    asm("fn (fid)(a: bits[i]) -> bits[1] { ret op_6_(aid): bits[1] = "    \
        "identity(a, pos=(loc)) }"                                        \
        : "=r"(ret)                                                       \
        : "i"(1), "parama"(__IMPL<std::max(ToW, Width), Signed | ToSign>::\
                      Operate(valB.storage, valA.storage)));              \
    return ret;                                                           \
  }

  COMPARISON_OP_WITH_SIGN(>, GreaterWithSign);
  COMPARISON_OP_WITH_SIGN(>=, GreaterOrEqWithSign);
  COMPARISON_OP_WITH_SIGN(<, LessWithSign);
  COMPARISON_OP_WITH_SIGN(<=, LessOrEqWithSign);

  inline XlsInt operator=(const XlsInt &o) {
    asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_7_(aid): bits[i] = "
        "identity(a, pos=(loc)) }"
        : "=r"(this->storage)
        : "i"(Width), "a"(o));
    return *this;
  }

  // Width+1 because we need to be able to represent shifting all digits
  typedef XlsInt<Log2Ceil<Width + 1>, false> index_t;

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
        : "i"(Width), "r"(ToW), "c"(index_t::width), "a"(this->storage),
          "o"(offset.storage));
    return ret;
  }

  inline bool operator[](index_t i) const {
    bool ret;
    asm("fn (fid)(a: bits[i], o: bits[c]) -> bits[1] { ret op_(aid): bits[1] = "
        "dynamic_bit_slice(a, o, width=1, pos=(loc)) }"
        : "=r"(ret)
        : "i"(Width), "c"(index_t::width), "a"(this->storage), "o"(i.storage));
    return ret;
  }

  // --- Hack: see comments for BitElemRef
  inline BitElemRef operator[](index_t i) {
    return BitElemRef(static_cast<bool>(slc<1>(i)));

    // NOP to ensure that clang parses the set_element functions
    set_element_bitref(0, 1);
    set_element_xlsint(0, 1);
    set_element_int(0, 1);
  }

  inline XlsInt set_element_bitref(index_t i, BitElemRef rvalue) {
    set_slc(i, XlsInt<1, false>(rvalue));
    return *this;
  }

  inline XlsInt set_element_int(index_t i, int rvalue) {
    set_slc(i, XlsInt<1, false>(rvalue));
    return *this;
  }

  inline XlsInt set_element_xlsint(index_t i, XlsInt<1, false> rvalue) {
    set_slc(i, rvalue);
    return *this;
  }

  // --- / Hack

  inline XlsInt reverse() const {
    XlsInt<Width, false> reverse_out;
    asm("fn (fid)(a: bits[i]) -> bits[i] { "
        "  ret (aid): bits[i] = reverse(a, pos=(loc)) }"
        : "=r"(reverse_out.storage)
        : "i"(Width), "a"(this->storage));
    return reverse_out;
  }

  inline index_t clz() const {
    XlsInt<Width, false> reverse_out;
    asm("fn (fid)(a: bits[i]) -> bits[i] { "
        "  ret (aid): bits[i] = reverse(a, pos=(loc)) }"
        : "=r"(reverse_out.storage)
        : "i"(Width), "a"(this->storage));

    XlsInt<Width + 1, false> one_hot_out;
    asm("fn (fid)(a: bits[i]) -> bits[c] { "
        "  ret (aid): bits[c] = one_hot(a, lsb_prio=true, pos=(loc)) }"
        : "=r"(one_hot_out.storage)
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
  XlsInt set_slc(index_t offset, XlsInt<ToW, ToSign> slice_raw) {
    static_assert(index_t::width > 0, "Bug in Log2Ceil");
    static_assert(Width >= ToW, "Can't set a slice wider than the source");
    static_assert(Width > 0, "Can't slice 0 bits");
    static_assert(ToW > 0, "Can't slice 0 bits");

    asm("fn (fid)(a: bits[i], o: bits[c], u: bits[d]) -> bits[i] { ret "
        "op_(aid): bits[i] = "
        "bit_slice_update(a, o, u, pos=(loc)) }"
        : "=r"(this->storage)
        : "i"(Width), "c"(index_t::width), "d"(ToW), "a"(this->storage),
          "o"(offset.storage), "u"(slice_raw.storage));

    return *this;
  }

  template <int ToW, bool ToSign>
  friend class XlsInt;
};

template <int Width, bool Signed>
inline std::ostream &operator<<(std::ostream &os,
                                const XlsInt<Width, Signed> &x) {
  return os;
}

#define ac_int XlsInt

// XLS[cc] doesn't support returning a reference
#undef ASSIGN_OP_WITH_INT
#define ASSIGN_OP_WITH_INT(ASSIGN_OP, C_TYPE, W2, S2)                    \
  template <int W, bool S>                                               \
  inline ac_int<W, S> operator ASSIGN_OP(ac_int<W, S> &op, C_TYPE op2) { \
    return op.operator ASSIGN_OP(ac_int<W2, S2>(op2));                   \
  }

OPS_WITH_INT(bool, 1, false)
OPS_WITH_INT(signed char, 8, true)
OPS_WITH_INT(unsigned char, 8, false)
OPS_WITH_INT(short, 16, true)
OPS_WITH_INT(unsigned short, 16, false)
OPS_WITH_INT(int, 32, true)
OPS_WITH_INT(unsigned int, 32, false)
OPS_WITH_INT(long, 64, true)
OPS_WITH_INT(unsigned long, 64, false)
OPS_WITH_INT(long long, 64, true)
OPS_WITH_INT(unsigned long long, 64, false)

#undef ac_int

#endif  // XLS_INT
