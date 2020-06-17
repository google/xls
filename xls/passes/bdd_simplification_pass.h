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

#ifndef XLS_PASSES_BDD_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_BDD_SIMPLIFICATION_PASS_H_

#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// Runs BDD-based simplifications on the function. Currently this is a very
// limited set of optimization including one-hot removal and replacement of
// statically known values with literals.
// TODO(meheff): Add more BDD-based optimizations.
class BddSimplificationPass : public FunctionPass {
 public:
  explicit BddSimplificationPass(bool split_ops)
      : FunctionPass("bdd_simp", "BDD-based Simplification"),
        split_ops_(split_ops) {}
  ~BddSimplificationPass() override {}

  // Run all registered passes in order of registration.
  xabsl::StatusOr<bool> RunOnFunction(Function* f, const PassOptions& options,
                                      PassResults* results) const override;

 private:
  bool split_ops_;
};

}  // namespace xls

#endif  // XLS_PASSES_BDD_SIMPLIFICATION_PASS_H_
