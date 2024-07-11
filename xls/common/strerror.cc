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

#include "xls/common/strerror.h"

#include <string.h>  // NOLINT needed for strerror_r

#include <string>
#include <type_traits>

#include "absl/strings/str_format.h"

namespace xls {

std::string Strerror(int error_num) {
  // The strerror_r function is available in two versions: XSI-compliant version
  // and a GNU-specific version. The GNU-specific version in glibc returns an
  // char* pointing to the error string. The XSI-compliant version returns an
  // int where 0 indicates success.
  // The second argument to strerror_r may be declared 'nonnull' in some
  // implementations, which can cause certain compilers to raise a warning,
  // even in a decltype() specifier. To avoid this, we pass in a non-null dummy
  // arg.
  using strerror_r_type =
      decltype(strerror_r(0, reinterpret_cast<char*>(1), 0));
  constexpr bool kXsiCompliant = std::is_same<strerror_r_type, int>::value;
  constexpr bool kGnuSpecific = std::is_same<strerror_r_type, char*>::value;
  static_assert(kXsiCompliant != kGnuSpecific,
                "strerror_r should be either the XSI-compliant version or the "
                "GNU-specific version");

  constexpr int kBufferSize = 512;
  char buffer[kBufferSize] = {};

  if constexpr (kXsiCompliant) {
    if (strerror_r(error_num, buffer, kBufferSize) !=
        static_cast<strerror_r_type>(0)) {  // NOLINT(modernize-use-nullptr):
      return absl::StrFormat(
          "Unknown error, strerror_r failed. error number %d", error_num);
    }
    return buffer;
  } else if constexpr (kGnuSpecific) {
    // Note that without the `reinterpret_cast`, this cannot be compiled with
    // the XSI-compliant version of strerror_r.
    return reinterpret_cast<char*>(strerror_r(error_num, buffer, kBufferSize));
  }
}

}  // namespace xls
