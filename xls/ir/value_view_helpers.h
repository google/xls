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

// This file holds a collection of helpful utilities when dealing with value
// views.
#ifndef XLS_IR_VALUE_VIEW_HELPERS_H_
#define XLS_IR_VALUE_VIEW_HELPERS_H_

#include "absl/base/casts.h"
#include "xls/ir/value_view.h"

namespace xls {

// View representation of a 32-bit float value.
using F32TupleView = TupleView<BitsView<1>, BitsView<8>, BitsView<23>>;
using PackedF32TupleView =
    PackedTupleView<PackedBitsView<1>, PackedBitsView<8>, PackedBitsView<23>>;

// Returns the flat float contained in the specified view tuple.
inline float F32TupleViewToFloat(F32TupleView tuple) {
  return absl::bit_cast<float>((tuple.Get<0>().GetValue() << 31) |
                               (tuple.Get<1>().GetValue() << 23) |
                               (tuple.Get<2>().GetValue() & 0x7FFFFF));
}

// Populates the specified view with the values from the input float.
inline void PopulateAsF32TupleView(float f, uint8_t* view_buffer) {
  using MutableF32TupleView =
      MutableTupleView<MutableBitsView<1>, MutableBitsView<8>,
                       MutableBitsView<23>>;
  MutableF32TupleView view(view_buffer);
  uint32_t i = absl::bit_cast<uint32_t>(f);
  view.Get<0>().SetValue(i >> 31);
  view.Get<1>().SetValue((i >> 23) & 0xff);
  view.Get<2>().SetValue(i & 0x7FFFFF);
}

}  // namespace xls

#endif  // XLS_IR_VALUE_VIEW_HELPERS_H_
