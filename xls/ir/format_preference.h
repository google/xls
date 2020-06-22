// Copyright 2020 Google LLC
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

#ifndef XLS_IR_FORMAT_PREFERENCE_H_
#define XLS_IR_FORMAT_PREFERENCE_H_

#include <ostream>

#include "absl/strings/string_view.h"

namespace xls {

// Explains what formatting technique should be used to convert a bit value into
// string form.
enum class FormatPreference {
  // Default formatting is decimal for values which fit in 64 bits. Otherwise
  // hexadecimal is used.
  kDefault,
  kBinary,
  kDecimal,
  kHex,
};

absl::string_view FormatPreferenceToString(FormatPreference preference);

inline std::ostream& operator<<(std::ostream& os, FormatPreference preference) {
  os << FormatPreferenceToString(preference);
  return os;
}

}  // namespace xls

#endif  // XLS_IR_FORMAT_PREFERENCE_H_
