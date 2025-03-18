// Copyright 2025 The XLS Authors
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

#ifndef XLS_IR_PARTIAL_OPS_H_
#define XLS_IR_PARTIAL_OPS_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xls/ir/partial_information.h"

namespace xls::partial_ops {

PartialInformation Join(PartialInformation a, const PartialInformation& b);
PartialInformation Meet(PartialInformation a, const PartialInformation& b);

PartialInformation Not(PartialInformation p);
PartialInformation And(PartialInformation a, const PartialInformation& b);
PartialInformation Or(PartialInformation a, const PartialInformation& b);
PartialInformation Xor(PartialInformation a, const PartialInformation& b);
PartialInformation Nand(PartialInformation a, const PartialInformation& b);
PartialInformation Nor(PartialInformation a, const PartialInformation& b);

PartialInformation AndReduce(const PartialInformation& p);
PartialInformation OrReduce(const PartialInformation& p);
PartialInformation XorReduce(const PartialInformation& p);

PartialInformation Concat(absl::Span<PartialInformation const> infos);
PartialInformation SignExtend(const PartialInformation& p, int64_t width);
PartialInformation ZeroExtend(const PartialInformation& p, int64_t width);
PartialInformation Truncate(const PartialInformation& p, int64_t width);
PartialInformation BitSlice(const PartialInformation& p, int64_t start,
                            int64_t width);

PartialInformation DynamicBitSlice(const PartialInformation& p,
                                   const PartialInformation& start,
                                   int64_t width);
PartialInformation BitSliceUpdate(const PartialInformation& to_update,
                                  const PartialInformation& start,
                                  const PartialInformation& update_value);

PartialInformation Reverse(PartialInformation p);

PartialInformation Decode(const PartialInformation& p, int64_t width);
PartialInformation Encode(const PartialInformation& p);

PartialInformation OneHotLsbToMsb(const PartialInformation& p);
PartialInformation OneHotMsbToLsb(const PartialInformation& p);

PartialInformation OneHotSelect(const PartialInformation& selector,
                                absl::Span<PartialInformation const> cases,
                                bool selector_can_be_zero);

PartialInformation Gate(const PartialInformation& control,
                        PartialInformation input);

PartialInformation Neg(PartialInformation p);
PartialInformation Add(PartialInformation a, const PartialInformation& b);
PartialInformation Sub(PartialInformation a, const PartialInformation& b);
PartialInformation UMul(const PartialInformation& a,
                        const PartialInformation& b, int64_t output_bitwidth);
PartialInformation UDiv(const PartialInformation& a,
                        const PartialInformation& b);
PartialInformation UMod(const PartialInformation& a,
                        const PartialInformation& b);
PartialInformation SMul(const PartialInformation& a,
                        const PartialInformation& b, int64_t output_bitwidth);
PartialInformation SDiv(const PartialInformation& a,
                        const PartialInformation& b);
PartialInformation SMod(const PartialInformation& a,
                        const PartialInformation& b);

PartialInformation Shll(PartialInformation a, const PartialInformation& b);
PartialInformation Shrl(PartialInformation a, const PartialInformation& b);
PartialInformation Shra(PartialInformation a, const PartialInformation& b);

PartialInformation Eq(const PartialInformation& a, const PartialInformation& b);
PartialInformation Ne(const PartialInformation& a, const PartialInformation& b);

PartialInformation ULt(const PartialInformation& a,
                       const PartialInformation& b);
inline PartialInformation UGt(const PartialInformation& a,
                              const PartialInformation& b) {
  return ULt(b, a);
}
inline PartialInformation ULe(const PartialInformation& a,
                              const PartialInformation& b) {
  return Not(ULt(b, a));
}
inline PartialInformation UGe(const PartialInformation& a,
                              const PartialInformation& b) {
  return Not(ULt(a, b));
}

PartialInformation SLt(const PartialInformation& a,
                       const PartialInformation& b);
inline PartialInformation SGt(const PartialInformation& a,
                              const PartialInformation& b) {
  return SLt(b, a);
}
inline PartialInformation SLe(const PartialInformation& a,
                              const PartialInformation& b) {
  return Not(SLt(b, a));
}
inline PartialInformation SGe(const PartialInformation& a,
                              const PartialInformation& b) {
  return Not(SLt(a, b));
}

// Bit ops.

}  // namespace xls::partial_ops

#endif  // XLS_IR_PARTIAL_OPS_H_
