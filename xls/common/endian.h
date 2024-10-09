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

#ifndef XLS_COMMON_ENDIAN_H_
#define XLS_COMMON_ENDIAN_H_

#include "absl/base/config.h"

namespace xls {

enum class Endianness {
  kLittleEndian,
  kBigEndian,
};

#if defined(ABSL_IS_BIG_ENDIAN)
inline constexpr Endianness kEndianness = Endianness::kBigEndian;
#elif defined(ABSL_IS_LITTLE_ENDIAN)
inline constexpr Endianness kEndianness = Endianness::kLittleEndian;
#else
#error "Cannot determine endianness"
#endif

}  // namespace xls

#endif  // XLS_COMMON_ENDIAN_H_
