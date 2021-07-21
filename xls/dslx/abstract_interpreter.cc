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

#include "xls/dslx/abstract_interpreter.h"

namespace xls::dslx {

AbstractInterpreter::ScopedTypeInfoSwap::ScopedTypeInfoSwap(
    AbstractInterpreter* interp, TypeInfo& updated)
    : interp(interp), original(XLS_DIE_IF_NULL(interp->GetCurrentTypeInfo())) {
  interp->SetCurrentTypeInfo(updated);
}

AbstractInterpreter::ScopedTypeInfoSwap::ScopedTypeInfoSwap(
    AbstractInterpreter* interp, Module* module)
    : ScopedTypeInfoSwap(interp, *interp->GetRootTypeInfo(module).value()) {}

AbstractInterpreter::ScopedTypeInfoSwap::ScopedTypeInfoSwap(
    AbstractInterpreter* interp, AstNode* node)
    : ScopedTypeInfoSwap(interp, node->owner()) {}

AbstractInterpreter::ScopedTypeInfoSwap::~ScopedTypeInfoSwap() {
  XLS_CHECK(original != nullptr);
  interp->SetCurrentTypeInfo(*original);
}

}  // namespace xls::dslx
