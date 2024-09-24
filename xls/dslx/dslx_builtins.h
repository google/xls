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

// Has the enumeration of all the various built-in functions and associated
// routines with that enumeration.

#ifndef XLS_DSLX_DSLX_BUILTINS_H_
#define XLS_DSLX_DSLX_BUILTINS_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"

namespace xls::dslx {

#define XLS_DSLX_BUILTIN_EACH(X)          \
  X("and_reduce", kAndReduce)             \
  X("array_rev", kArrayRev)               \
  X("array_size", kArraySize)             \
  X("assert_eq", kAssertEq)               \
  X("assert_lt", kAssertLt)               \
  X("bit_slice_update", kBitSliceUpdate)  \
  X("checked_cast", kCheckedCast)         \
  X("clz", kClz)                          \
  X("cover!", kCover)                     \
  X("ctz", kCtz)                          \
  X("gate!", kGate)                       \
  X("enumerate", kEnumerate)              \
  X("fail!", kFail)                       \
  X("assert!", kAssert)                   \
  X("map", kMap)                          \
  X("decode", kDecode)                    \
  X("encode", kEncode)                    \
  X("one_hot", kOneHot)                   \
  X("one_hot_sel", kOneHotSel)            \
  X("or_reduce", kOrReduce)               \
  X("priority_sel", kPriorityhSel)        \
  X("range", kRange)                      \
  X("rev", kRev)                          \
  X("zip", kZip)                          \
  X("widening_cast", kWideningCast)       \
  X("signex", kSignex)                    \
  X("smulp", kSMulp)                      \
  X("slice", kSlice)                      \
  X("trace!", kTrace)                     \
  X("umulp", kUMulp)                      \
  X("update", kUpdate)                    \
  X("xor_reduce", kXorReduce)             \
  X("join", kJoin)                        \
  X("token", kToken)                      \
  /* send/recv routines */                \
  X("send", kSend)                        \
  X("send_if", kSendIf)                   \
  X("recv", kRecv)                        \
  X("recv_if", kRecvIf)                   \
  X("recv_nonblocking", kRecvNonBlocking) \
  X("recv_if_nonblocking", kRecvIfNonBlocking)

// Enum that represents all the DSLX builtin functions.
//
// Functions can be held in values, either as user defined ones or builtin ones
// (represented via this enumerated value).
enum class Builtin : uint8_t {
#define ENUMIFY(__str, __enum, ...) __enum,
  XLS_DSLX_BUILTIN_EACH(ENUMIFY)
#undef ENUMIFY
};

absl::StatusOr<Builtin> BuiltinFromString(std::string_view name);

std::string BuiltinToString(Builtin builtin);

inline constexpr Builtin kAllBuiltins[] = {
#define ELEMIFY(__str, __enum, ...) Builtin::__enum,
    XLS_DSLX_BUILTIN_EACH(ELEMIFY)
#undef ELEMIFY
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_DSLX_BUILTINS_H_
