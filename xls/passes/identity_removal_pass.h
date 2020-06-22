// Copyright 2020 Google LLC
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

// Identity Removal - eliminate all identity() expressions.

#ifndef XLS_PASSES_IDENTITY_REMOVAL_PASS_H_
#define XLS_PASSES_IDENTITY_REMOVAL_PASS_H_

#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// class IdentityRemovalPass eliminates all identity() expressions
// by forward substituting it's parameters to the uses of the
// identity's def.
class IdentityRemovalPass : public FunctionPass {
 public:
  IdentityRemovalPass() : FunctionPass("ident_remove", "Identity Removal") {}
  ~IdentityRemovalPass() override {}

  // Iterate all nodes and eliminate identities.
  xabsl::StatusOr<bool> RunOnFunction(Function* f, const PassOptions& options,
                                      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_IDENTITY_REMOVAL_PASS_H_
