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

#ifndef XLS_CONTRIB_XLSCC_SYNTH_ONLY_XLS_BUILTIN_H_
#define XLS_CONTRIB_XLSCC_SYNTH_ONLY_XLS_BUILTIN_H_

#ifdef __XLSCC_SYNTH_COMPAT__
#ifndef __SYNTHESIS__
#define __SYNTHESIS__ 1
#endif

#ifndef HLS_STATIC
#define HLS_STATIC static
#endif

#ifndef HLS_PRAGMA
#define HLS_PRAGMA(x) _Pragma(#x)
#endif

#ifndef XLS_ASSERT
#define XLS_ASSERT(x) ((void)0)
#endif

#ifndef assert
#define assert(...) ((void)0)
#endif

#ifndef NULL
#define NULL nullptr
#endif

using size_t = decltype(sizeof(0));
using ptrdiff_t =
    decltype(static_cast<char*>(nullptr) - static_cast<char*>(nullptr));
using int8_t = signed char;
using uint8_t = unsigned char;
using int16_t = short;
using uint16_t = unsigned short;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = signed long;
using uint64_t = unsigned long;

namespace std {
using ::int16_t;
using ::int32_t;
using ::int64_t;
using ::int8_t;
using ::ptrdiff_t;
using ::size_t;
using ::uint16_t;
using ::uint32_t;
using ::uint64_t;
using ::uint8_t;

#ifndef XLS_CONTRIB_XLSCC_SYNTH_ONLY_INITIALIZER_LIST_
#define XLS_CONTRIB_XLSCC_SYNTH_ONLY_INITIALIZER_LIST_
template <typename E>
class initializer_list {
 public:
  using value_type = E;
  using reference = const E&;
  using const_reference = const E&;
  using size_type = size_t;
  using iterator = const E*;
  using const_iterator = const E*;

  constexpr initializer_list() noexcept : array_(nullptr), len_(0) {}
  constexpr size_t size() const noexcept { return len_; }
  constexpr const E* begin() const noexcept { return array_; }
  constexpr const E* end() const noexcept { return array_ + len_; }

 private:
  const E* array_;
  size_t len_;
};
#endif  // XLS_CONTRIB_XLSCC_SYNTH_ONLY_INITIALIZER_LIST_
}  // namespace std
#endif  // __XLSCC_SYNTH_COMPAT__

template <int N>
struct __xls_bits {};

struct __xls_token_raw {};

class [[hls_no_tuple]] __xls_token {
 public:
  inline __xls_token() {}

  inline __xls_token(const __xls_token& o) {
    asm("fn (fid)(a: token) -> token { ret op(aid): token = "
        "identity(a, pos=(loc)) }"
        : "=r"(storage)
        : "a"(o.storage));
  }

  inline __xls_token operator=(const __xls_token& o) {
    asm("fn (fid)(a: token) -> token { ret op(aid): token = "
        "identity(a, pos=(loc)) }"
        : "=r"(storage)
        : "a"(o.storage));

    return *this;
  }

 private:
  __xls_token_raw storage;
};

// Should match OpType
enum __xls_channel_dir {
  __xls_channel_dir_Unknown = 0,  // OpType::kNull
  __xls_channel_dir_Out = 1,      // OpType::kSend
  __xls_channel_dir_In = 2,       // OpType::kRecv
  __xls_channel_dir_InOut = 3     // OpType::kSendRecv
};

template <typename T, __xls_channel_dir Dir = __xls_channel_dir_Unknown>
class __xls_channel {
 public:
  class TokenizedValue {
   public:
    T value() const { return value_; }
    __xls_token token() const { return token_; }

   private:
    T value_;
    __xls_token token_;
  };

  // This separate method is necessary because C++'s implicit conversions
  // don't allow TokenizedValue to substitute for T as a return in all cases.
  TokenizedValue read_with_token(__xls_token token_in = __xls_token()) const {
    return TokenizedValue();
  }
  T read(__xls_token token_in = __xls_token()) const { return T(); }
  __xls_token write(T val, __xls_token token_in = __xls_token()) const {
    return __xls_token();
  }
  void read(T& out) const { (void)out; }
  bool nb_read(T& out) const {
    (void)out;
    return true;
  }
  bool nb_write(const T& val) const {
    (void)val;
    return true;
  }
  bool empty() const { return false; }
  unsigned int size() const { return 0; }
  bool available(unsigned int k = 1) const {
    (void)k;
    return true;
  }
};

// Size is an int so that templates declaring `int N` and aliases such as
// `template <typename T, int N> using Memory = __xls_memory<T, N>;` can deduce
// N from __xls_memory arguments.
template <typename T, int Size>
class __xls_memory {
 public:
  using value_type = T;
  using element_type = T;

  constexpr unsigned long long size() const { return Size; }

  T& operator[](long long int addr) const {
    static T ret;
    return ret;
  }
  void write(long long int addr, const T& value) const { return; }
  T read(long long int addr) const { return T(); }
};

// Bypass no outputs error
int __xlscc_unimplemented() { return 0; }

void __xlscc_assert(const char* message, bool condition,
                    const char* label = nullptr, ...) {}

// See XLS IR trace op format
void __xlscc_trace(const char* fmt, ...) {}

class __xlscc_trace_stream_type {
 public:
  template <typename T>
  __xlscc_trace_stream_type& operator<<(const T& val) {
    return *this;
  }
};

__xlscc_trace_stream_type __xlscc_trace_stream;

// Forces the FSM to a new activation / initiation.
// The transition will be conditional on the context like an IO operation.
template <bool conditional = false>
void __xlscc_activation_barrier(bool cond = true) {}

bool __xlscc_on_reset = false;

// Returns bits for 32.32 fixed point representation
__xls_bits<64> __xlscc_fixed_32_32_bits_for_double(double input);
__xls_bits<64> __xlscc_fixed_32_32_bits_for_float(float input);

#endif  // XLS_CONTRIB_XLSCC_SYNTH_ONLY_XLS_BUILTIN_H_
