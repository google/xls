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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_BUILTIN_TRAIT_DERIVER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_BUILTIN_TRAIT_DERIVER_H_

#include <memory>

#include "xls/dslx/type_system_v2/trait_deriver.h"

namespace xls::dslx {

// Creates the `TraitDeriver` that should be used for builtin traits. These
// traits are declared in `builtin_stubs.x`.
std::unique_ptr<TraitDeriver> CreateBuiltinTraitDeriver();

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_BUILTIN_TRAIT_DERIVER_H_
