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

#ifndef XLS_CONTRIB_MLIR_UTIL_CONVERSION_UTILS_H_
#define XLS_CONTRIB_MLIR_UTIL_CONVERSION_UTILS_H_

#include "llvm/include/llvm/ADT/APFloat.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace mlir::xls {

// Conversion utilities between XLS and MLIR value types.

// Converts an APInt to an XLS Bits. This always succeeds.
::xls::Bits bitsFromAPInt(llvm::APInt apInt);

// Converts an XLS Bits to an APInt. This always succeeds.
llvm::APInt bitsToAPInt(::xls::Bits bits);

// Converts an APFloat to an XLS Value tuple of (sign, exponent, mantissa). This
// always succeeds.
::xls::Value tupleFromAPFloat(llvm::APFloat apFloat);

// Converts an XLS Value tuple of (sign, exponent, mantissa) to an APFloat. This
// always succeeds on valid inputs and is undefined otherwise.
llvm::APFloat tupleToAPFloat(::xls::Value tuple, FloatType type);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_UTIL_CONVERSION_UTILS_H_
