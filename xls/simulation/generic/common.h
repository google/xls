// Copyright 2023 The XLS Authors
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

#ifndef XLS_SIMULATION_GENERIC_COMMON_H_
#define XLS_SIMULATION_GENERIC_COMMON_H_

#include <cstdint>

#include "xls/common/logging/logging.h"

enum class AccessWidth : uint8_t { BYTE, WORD, DWORD, QWORD };

enum class IRQEnum : uint8_t { NoChange, SetIRQ, UnsetIRQ };

inline uint64_t BytesToBits(uint64_t byte_count) {
  XLS_DCHECK_GE(byte_count, 0);
  XLS_DCHECK_LE(byte_count, 8);
  return byte_count << 3;
}

inline uint64_t BitsToBytes(uint64_t bit_count) {
  return (bit_count & 0x7) ? (bit_count >> 3) + 1 : (bit_count >> 3);
}

#endif  // XLS_SIMULATION_GENERIC_COMMON_H_
