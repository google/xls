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

// This file holds a collection of helpful utilities when dealing with value
// views.
#ifndef XLS_IR_VALUE_VIEW_HELPERS_H_
#define XLS_IR_VALUE_VIEW_HELPERS_H_

#include "absl/base/casts.h"
#include "xls/ir/value_view.h"

namespace xls {

// View representation of a 32-bit float value.
using F32TupleView = TupleView<BitsView<1>, BitsView<8>, BitsView<23>>;

inline float F32TupleViewToFloat(F32TupleView tuple) {
  return absl::bit_cast<float>((tuple.Get<0>().GetValue() << 31) |
                               (tuple.Get<1>().GetValue() << 23) |
                               (tuple.Get<2>().GetValue() & 0x7FFFFF));
}

}  // namespace xls

#endif  // XLS_IR_VALUE_VIEW_HELPERS_H_
