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

#ifndef __STDINT_H__
#define __STDINT_H__

// See Translator::TranslateTypeFromClang()
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef signed short int16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef unsigned long uint64_t;
typedef signed long int64_t;
typedef unsigned long long int uintmax_t;

static_assert(sizeof(int8_t) == 1);
static_assert(sizeof(uint8_t) == 1);
static_assert(sizeof(int16_t) == 2);
static_assert(sizeof(uint16_t) == 2);
static_assert(sizeof(int32_t) == 4);
static_assert(sizeof(uint32_t) == 4);
static_assert(sizeof(int64_t) == 8);
static_assert(sizeof(uint64_t) == 8);

#define INT8_MAX 127
#define INT8_MIN -128
#define UINT8_MAX 255

#define INT16_MAX 32767
#define INT16_MIN -32768
#define UINT16_MAX 65535

#define INT32_MAX 2147483647
#define INT32_MIN -2147483648
#define UINT32_MAX 4294967295

#define INT64_MAX 9223372036854775807L
#define INT64_MIN -9223372036854775808L
#define UINT64_MAX 18446744073709551615UL

#endif  // __STDINT_H__
