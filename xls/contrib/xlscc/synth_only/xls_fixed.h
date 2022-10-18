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

#include <cstdint>

#include "xls_int.h"

// Non-functional stub
template <int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
class XlsFixed {
 public:
  XlsFixed() { (void)__xlscc_unimplemented(); }
  template <typename T>
  XlsFixed(const T& o) {
    (void)__xlscc_unimplemented();
  }
  int to_int() const {
    (void)__xlscc_unimplemented();
    return 0;
  }
  XlsInt<AC_MAX(IntegerWidth, 1), Signed> to_ac_int() const {
    (void)__xlscc_unimplemented();
    return 0;
  }
};

// Minimal functionality for saturation usage
template <int Width, bool Signed, ac_datatypes::ac_q_mode Quantization>
class XlsFixed<Width, Width, Signed, Quantization, ac_datatypes::AC_SAT> {
  // TODO(seanhaskell): Use XlsInt instead of int64_t to allow for arbitrary
  // widths
  static_assert(Width <= 63);

 public:
  XlsFixed() : val_(0) {}
  XlsFixed(int64_t val) : val_(clamp(val)) {}
  template <int Width2, bool Signed2>
  XlsFixed(XlsInt<Width2, Signed2> val) : val_(clamp(val)) {
    static_assert(Width2 <= 63);
  }
  int64_t to_int() const { return val_; }
  XlsInt<Width, Signed> to_ac_int() const { return val_; }

 private:
  static int64_t clamp(int64_t val) {
    int64_t ret;
    constexpr int64_t min_val =
        (!Signed) ? 0L : (-(1L << (int64_t(Width) - 1L)));
    constexpr int64_t max_val = (!Signed)
                                    ? ((1L << int64_t(Width)) - 1L)
                                    : ((1L << (int64_t(Width) - 1L)) - 1L);
    if (val < min_val) {
      ret = min_val;
    } else if (val > max_val) {
      ret = max_val;
    } else {
      ret = val;
    }
    return ret;
  }

  XlsInt<Width, Signed> val_;
};

template<typename O, int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
inline O operator+(const XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow>& o, const O &op) {
  (void)__xlscc_unimplemented();
  return O();
}

template<typename O, int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
inline O operator+(const O &op, const XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow>& o) {
  (void)__xlscc_unimplemented();
  return O();
}


template<typename O, int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
inline O operator-(const XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow>& o, const O &op) {
  (void)__xlscc_unimplemented();
  return O();
}

template<typename O, int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
inline O operator-(const O &op, const XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow>& o) {
  (void)__xlscc_unimplemented();
  return O();
}

template<typename O, int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
inline O operator*(const XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow>& o, const O &op) {
  (void)__xlscc_unimplemented();
  return O();
}

template<typename O, int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization,
          ac_datatypes::ac_o_mode Overflow>
inline O operator*(const O &op, const XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow>& o) {
  (void)__xlscc_unimplemented();
  return O();
}

#endif  // XLS_FIXED_H
