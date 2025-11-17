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

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/type.h"

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TRAIT_DERIVER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TRAIT_DERIVER_H_

namespace xls::dslx {

// An object that can be plugged into type inference to generate standard trait
// implementations.
class TraitDeriver {
 public:
  virtual ~TraitDeriver() = default;

  // Generates a body of `function` from `trait` for the given `struct_def`.
  virtual absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const StructType& struct_type, const Function& function) = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TRAIT_DERIVER_H_
