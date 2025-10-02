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

#include <cassert>
#include <cstdint>
#include <utility>

#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/APFloat.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "xls/common/math_util.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace mlir::xls {
using ::llvm::APFloat;
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::FloatType;
using ::xls::Bits;
using ::xls::CeilOfRatio;
using ::xls::InlineBitmap;
using ::xls::Value;

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

Value tupleFromAPFloat(APFloat apFloat) {
  // Determine the number of bits in the mantissa, exponent, and sign bit.
  int32_t totalNumBits = APFloat::semanticsSizeInBits(apFloat.getSemantics());
  int32_t mantNumBits = APFloat::semanticsPrecision(apFloat.getSemantics()) - 1;
  bool isSigned = APFloat::semanticsHasSignedRepr(apFloat.getSemantics());
  assert(APFloat::hasSignBitInMSB(apFloat.getSemantics()) &&
         "Expected sign bit in MSB");
  int32_t expNumBits = totalNumBits - mantNumBits - (isSigned ? 1 : 0);

  // Extract the sign bit, mantissa, and exponent from the APFloat.
  APInt intBits = apFloat.bitcastToAPInt();
  APInt sign = intBits.extractBits(1, expNumBits + mantNumBits);
  APInt exp = intBits.extractBits(expNumBits, mantNumBits);
  APInt mant = intBits.extractBits(mantNumBits, 0);

  // Convert the extracted bits to a tuple of XLS Bits.
  Bits signBits = bitsFromAPInt(sign);
  Bits expBits = bitsFromAPInt(exp);
  Bits mantBits = bitsFromAPInt(mant);
  return Value::Tuple({Value(signBits), Value(expBits), Value(mantBits)});
}

APFloat tupleToAPFloat(Value tuple, FloatType type) {
  // Extract the sign, exponent, and mantissa from the tuple.
  assert(tuple.IsTuple() && tuple.size() == 3 &&
         "Expected float value tuple of (sign, exponent, mantissa)");
  Bits signBits = tuple.element(0).bits();
  Bits expBits = tuple.element(1).bits();
  Bits mantBits = tuple.element(2).bits();

  // Convert the extracted bits to an APFloat.
  APInt sign = bitsToAPInt(signBits);
  APInt exp = bitsToAPInt(expBits);
  APInt mant = bitsToAPInt(mantBits);
  APInt intBits = sign.concat(exp).concat(mant);
  return APFloat(type.getFloatSemantics(), intBits);
}

}  // namespace mlir::xls
