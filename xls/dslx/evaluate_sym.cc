// Copyright 2021 The XLS Authors
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

#include "xls/dslx/evaluate_sym.h"

namespace xls::dslx {

#define DISPATCH_DEF(__expr_type)                                              \
  absl::StatusOr<InterpValue> Evaluate##__expr_typeSym(                        \
      __expr_type* expr, InterpBindings* bindings, ConcreteType* type_context, \
      AbstractInterpreter* interp) {                                           \
    return Evaluate##__expr_type(expr, bindings, type_context, interp);        \
  }

DISPATCH_DEF(Array)
DISPATCH_DEF(Attr)
DISPATCH_DEF(Binop)
DISPATCH_DEF(Carry)
DISPATCH_DEF(Cast)
DISPATCH_DEF(ColonRef)
DISPATCH_DEF(ConstRef)
DISPATCH_DEF(For)
DISPATCH_DEF(Index)
DISPATCH_DEF(Let)
DISPATCH_DEF(Match)
DISPATCH_DEF(NameRef)
DISPATCH_DEF(Number)
DISPATCH_DEF(SplatStructInstance)
DISPATCH_DEF(String)
DISPATCH_DEF(StructInstance)
DISPATCH_DEF(Ternary)
DISPATCH_DEF(Unop)
DISPATCH_DEF(While)
DISPATCH_DEF(XlsTuple)

#undef DISPATCH_DEF
}  // namespace xls::dslx
