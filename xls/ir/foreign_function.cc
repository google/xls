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

#include "xls/ir/foreign_function.h"

#include <optional>
#include <string>
#include <string_view>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/code_template.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/value.h"

namespace xls {

absl::StatusOr<ForeignFunctionData> ForeignFunctionDataCreateFromTemplate(
    std::string_view annotation) {
  absl::StatusOr<CodeTemplate> parse_result = CodeTemplate::Create(annotation);
  if (!parse_result.ok()) {
    return parse_result.status();
  }
  // We just pass the template along as string, but we validated it worked
  ForeignFunctionData result;
  result.set_code_template(annotation);
  return result;
}

void FfiPartialValueSubstituteHelper::SetNamedValue(std::string_view name,
                                                    const Value& value) {
  if (!value.IsBits()) {
    return;  // Only interested in scalars right now
  }
  // Emit these in hex to be interpreted as whatever it needs to be in the
  // Verilog substitution. This is somewhat breaking abstraction as we already
  // here decide on a representation that we know further down the generated
  // code needs. But then again: we already broke that by introducing
  // Verilog-specific templates.
  substitutions_[name] = absl::StrFormat(
      "%d'h%s", value.bits().bit_count(),
      BitsToRawDigits(value.bits(), FormatPreference::kPlainHex, true));
}

std::optional<::xls::ForeignFunctionData>
FfiPartialValueSubstituteHelper::GetUpdatedFfiData() const {
  if (!ffi_.has_value() || substitutions_.empty()) {
    return ffi_;  // Nothing to do.
  }
  absl::StatusOr<CodeTemplate> code_template =
      CodeTemplate::Create(ffi_->code_template());
  CHECK(code_template.ok()) << "unexpected: invalid template";
  std::string modified_template = code_template->Substitute(
      [&](std::string_view name) {
        auto found = substitutions_.find(name);
        if (found == substitutions_.end()) {
          return absl::StrCat("{", name, "}");  // Pass on original name
        }
        return found->second;
      },
      CodeTemplate::Escaped::kKeep);
  ForeignFunctionData result;
  result.set_code_template(modified_template);
  return result;
}

}  // namespace xls
