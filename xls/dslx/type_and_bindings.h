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

#ifndef XLS_DSLX_TYPE_AND_BINDINGS_H_
#define XLS_DSLX_TYPE_AND_BINDINGS_H_

#include "xls/dslx/concrete_type.h"
#include "xls/dslx/symbolic_bindings.h"

namespace xls::dslx {

struct TypeAndBindings {
  std::unique_ptr<ConcreteType> type;
  SymbolicBindings symbolic_bindings;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_AND_BINDINGS_H_
