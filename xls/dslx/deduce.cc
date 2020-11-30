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

#include "xls/dslx/deduce.h"

namespace xls::dslx {

absl::Status CheckBitwidth(const Number& number, const ConcreteType& type) {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bits_dim, type.GetTotalBitCount());
  XLS_RET_CHECK(absl::holds_alternative<int64>(bits_dim.value()));
  int64 bit_count = absl::get<int64>(bits_dim.value());
  absl::StatusOr<Bits> bits = number.GetBits(bit_count);
  if (!bits.ok()) {
    return absl::InternalError(
        absl::StrFormat("TypeInferenceError: %s %s Value '%s' does not fit in "
                        "the bitwidth of a %s (%d)",
                        number.span().ToString(), type.ToString(),
                        number.text(), type.ToString(), bit_count));
  }
  return absl::OkStatus();
}

}  // namespace xls::dslx
