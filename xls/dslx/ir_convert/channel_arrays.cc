// Copyright 2025 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with a copy of the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/dslx/ir_convert/channel_arrays.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

namespace {

constexpr std::string_view kBetweenDimsSeparator = "_";

}  // namespace

absl::StatusOr<std::vector<std::string>>
ChannelArrays::CreateAllArrayElementSuffixes(
    const std::vector<Expr*>& dims) const {
  std::vector<std::string> strings;
  // Note: dims are in the opposite of indexing order, and here we want to use
  // them to produce index strings in indexing order, hence the backwards loop.
  for (int64_t i = dims.size() - 1; i >= 0; --i) {
    Expr* dim = dims[i];
    XLS_ASSIGN_OR_RETURN(InterpValue dim_interp_value,
                         ConstexprEvaluator::EvaluateToValue(
                             import_data_, type_info_,
                             /*warning_collector=*/nullptr, bindings_, dim));
    XLS_ASSIGN_OR_RETURN(int64_t dim_value,
                         dim_interp_value.GetBitValueUnsigned());
    std::vector<std::string> new_strings;
    new_strings.reserve(dim_value * strings.size());
    for (int64_t element_index = 0; element_index < dim_value;
         element_index++) {
      if (strings.empty()) {
        new_strings.push_back(absl::StrCat(element_index));
        continue;
      }
      for (const std::string& next : strings) {
        new_strings.push_back(
            absl::StrCat(next, kBetweenDimsSeparator, element_index));
      }
    }
    strings = std::move(new_strings);
  }
  return strings;
}

}  // namespace xls::dslx
