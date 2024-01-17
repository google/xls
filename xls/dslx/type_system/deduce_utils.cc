// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system/deduce_utils.h"

#include <cstdint>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

absl::Status TryEnsureFitsInType(const Number& number, const BitsType& type) {
  XLS_VLOG(5) << "TryEnsureFitsInType; number: " << number.ToString() << " @ "
              << number.span() << " type: " << type;

  // Characters have a `u8` type. They can support the dash (negation symbol).
  if (number.number_kind() != NumberKind::kCharacter &&
      number.text()[0] == '-' && !type.is_signed()) {
    return TypeInferenceErrorStatus(
        number.span(), &type,
        absl::StrFormat("Number %s invalid: "
                        "can't assign a negative value to an unsigned type.",
                        number.ToString()));
  }

  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bits_dim, type.GetTotalBitCount());
  if (!std::holds_alternative<InterpValue>(bits_dim.value())) {
    // We have to wait for the dimension to be fully resolved before we can
    // check that the number is compliant.
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_dim.GetAsInt64());

  // Helper to give an informative error on the appropriate range when we
  // determine the numerical value given doesn't fit into the type.
  auto does_not_fit = [&number, &type, bit_count]() {
    std::string low;
    std::string high;
    if (type.is_signed()) {
      low = BitsToString(Bits::MinSigned(bit_count),
                         FormatPreference::kSignedDecimal);
      high = BitsToString(Bits::MaxSigned(bit_count),
                          FormatPreference::kSignedDecimal);
    } else {
      low = BitsToString(Bits(bit_count), FormatPreference::kUnsignedDecimal);
      high = BitsToString(Bits::AllOnes(bit_count),
                          FormatPreference::kUnsignedDecimal);
    }

    return TypeInferenceErrorStatus(
        number.span(), &type,
        absl::StrFormat("Value '%s' does not fit in "
                        "the bitwidth of a %s (%d). "
                        "Valid values are [%s, %s].",
                        number.text(), type.ToString(), bit_count, low, high));
  };

  XLS_ASSIGN_OR_RETURN(bool fits_in_type, number.FitsInType(bit_count));
  if (!fits_in_type) {
    return does_not_fit();
  }
  return absl::OkStatus();
}

void UseImplicitToken(DeduceCtx* ctx) {
  XLS_CHECK(!ctx->fn_stack().empty());
  Function* caller = ctx->fn_stack().back().f();
  // Note: caller could be nullptr; e.g. when we're calling a function that
  // can fail!() from the top level of a module; e.g. in a module-level const
  // expression.
  if (caller != nullptr) {
    ctx->type_info()->NoteRequiresImplicitToken(caller, true);
  }

  // TODO(rspringer): 2021-09-01: How to fail! from inside a proc?
}

bool IsNameRefTo(const Expr* e, const NameDef* name_def) {
  if (auto* name_ref = dynamic_cast<const NameRef*>(e)) {
    const AnyNameDef any_name_def = name_ref->name_def();
    return std::holds_alternative<const NameDef*>(any_name_def) &&
           std::get<const NameDef*>(any_name_def) == name_def;
  }
  return false;
}

}  // namespace xls::dslx
