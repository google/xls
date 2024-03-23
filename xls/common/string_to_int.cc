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

#include "xls/common/string_to_int.h"

#include <limits>
#include <string_view>

#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

absl::StatusOr<int64_t> StrTo64Base(std::string_view s, int base) {
  if (s == "0" || s == "-0") {
    return 0;
  }
  const std::string_view original = s;
  bool negated = absl::ConsumePrefix(&s, "-");

  if (base == 0) {
    if (absl::ConsumePrefix(&s, "0b")) {
      base = 2;
    } else if (absl::ConsumePrefix(&s, "0x")) {
      base = 16;
    } else if (absl::ConsumePrefix(&s, "0")) {
      base = 8;
    } else {
      base = 10;
    }
  } else {
    // When a base is provided, we try to consume the leading prefix.
    //
    // In octal, leading zeros are indistinguishable from the base indicator,
    // whereas in other bases there's no problem. But because we're lopping off
    // a leading zero, shouldn't matter for octal either.
    if (base == 16) {
      absl::ConsumePrefix(&s, "0x");
    } else if (base == 8) {
      absl::ConsumePrefix(&s, "0");
    } else if (base == 2) {
      absl::ConsumePrefix(&s, "0b");
    }
  }

  XLS_RET_CHECK_GE(base, 2);

  if (s.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Number contained no digits: \"%s\"", original));
  }

  auto char_to_digit = [&](char c) -> absl::StatusOr<uint64_t> {
    auto error = [&] {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Number character '%c' is invalid for numeric base %d in \"%s\"", c,
          base, original));
    };

    uint64_t digit = std::numeric_limits<uint64_t>::max();
    switch (base) {
      case 2:
      case 8:
      case 10:
        digit = c - '0';
        break;
      case 16: {
        if ('a' <= c && c <= 'f') {
          digit = c - 'a' + 0xa;
        } else if ('A' <= c && c <= 'F') {
          digit = c - 'A' + 0xa;
        } else if ('0' <= c && c <= '9') {
          digit = c - '0';
        } else {
          return error();
        }
        break;
      }
      default:
        LOG(FATAL) << "Invalid base: " << base;
    }
    if (digit >= base) {
      return error();
    }
    return digit;
  };
  absl::uint128 accum = 0;
  while (!s.empty()) {
    char c = s[0];
    s.remove_prefix(1);
    XLS_ASSIGN_OR_RETURN(uint64_t digit, char_to_digit(c));
    accum *= base;
    accum += digit;
    if (Uint128High64(accum) != 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Number overflows 64-bit integer: \"%s\"", original));
    }
  }
  // Note: even though the most-negative number in the space overflows when you
  // negate it (that is, `-std::numeric_limits<int64_t>::min()`), we consider
  // this to be not-an-error, because the characters we scanned didn't take us
  // past the 64-bit boundary.
  if (negated) {
    accum = -accum;
  }
  return Uint128Low64(accum);
}

}  // namespace xls
