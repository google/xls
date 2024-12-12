// Copyright 2024 The XLS Authors
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

#ifndef XLS_PUBLIC_C_API_FORMAT_PREFERENCE_H_
#define XLS_PUBLIC_C_API_FORMAT_PREFERENCE_H_

#include <stdint.h>  // NOLINT(modernize-deprecated-headers)

extern "C" {

// Note: We define the format preference enum with a fixed width integer type
// for clarity of the exposed ABI.
typedef int32_t xls_format_preference;
enum {
  xls_format_preference_default,
  xls_format_preference_binary,
  xls_format_preference_signed_decimal,
  xls_format_preference_unsigned_decimal,
  xls_format_preference_hex,
  xls_format_preference_plain_binary,
  xls_format_preference_plain_hex,
  xls_format_preference_zero_padded_binary,
  xls_format_preference_zero_padded_hex,
};

}  // extern "C"

#endif  // XLS_PUBLIC_C_API_FORMAT_PREFERENCE_H_
