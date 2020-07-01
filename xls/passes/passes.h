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

#ifndef XLS_PASSES_PASSES_H_
#define XLS_PASSES_PASSES_H_

#include <stdio.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/time/time.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Defines the pass types for passes which operate strictly on XLS IR (i.e.,
// xls::Package).
using Pass = PassBase<Package>;
using CompoundPass = CompoundPassBase<Package>;
using FixedPointCompoundPass = FixedPointCompoundPassBase<Package>;
using InvariantChecker = CompoundPass::InvariantChecker;

// Abstract base class for passes operate at function scope. The derived class
// must define RunOnFunction.
class FunctionPass : public Pass {
 public:
  FunctionPass(absl::string_view short_name, absl::string_view long_name)
      : Pass(short_name, long_name) {}

  virtual xabsl::StatusOr<bool> RunOnFunction(Function* f,
                                              const PassOptions& options,
                                              PassResults* results) const = 0;

  // Iterates over each function in the package calling RunOnFunction.
  xabsl::StatusOr<bool> Run(Package* p, const PassOptions& options,
                            PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_PASSES_H_
