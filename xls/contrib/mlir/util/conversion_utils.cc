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

#include "xls/contrib/mlir/util/conversion_utils.h"

#include <cstdint>
#include <utility>

#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "xls/common/math_util.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bits.h"

namespace mlir::xls {
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::xls::Bits;
using ::xls::CeilOfRatio;
using ::xls::InlineBitmap;

Bits bitsFromAPInt(APInt apInt) {
  // Create an InlinedBitmap with the raw data from the APInt. This works
  // because both APInt and InlinedBitmap internally use uint64_t[] for storage.
  //
  // This uses an implementation detail of APInt and InlinedBitmap, and is
  // tested in the unit tests.
  int64_t bitCount = apInt.getBitWidth();
  auto span =
      absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(apInt.getRawData()),
                          apInt.getNumWords() * sizeof(uint64_t));
  InlineBitmap bitmap =
      InlineBitmap::FromBytes(bitCount, span.first(CeilOfRatio(bitCount, 8L)));
  return Bits::FromBitmap(std::move(bitmap));
}

APInt bitsToAPInt(::xls::Bits bits) {
  if (bits.bit_count() <= 64) {
    return APInt(bits.bit_count(), bits.bitmap().GetWord(0));
  }

  // Ideally InlineBitmap would expose the words as an absl::Span, but it
  // doesn't, so we copy to a SmallVector.
  const InlineBitmap& bitmap = bits.bitmap();
  SmallVector<uint64_t> words(bitmap.word_count());
  for (int i = 0, e = bitmap.word_count(); i < e; ++i) {
    words[i] = bits.bitmap().GetWord(i);
  }
  return APInt(bits.bit_count(), ArrayRef(words));
}

}  // namespace mlir::xls
