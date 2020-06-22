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

// IR Canonicalization.

#ifndef XLS_PASSES_CANONICALIZATION_PASS_H_
#define XLS_PASSES_CANONICALIZATION_PASS_H_

#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// class CanonicalizationPass iterates over nodes and tries
// to canonicalize the expressions found. For example, for an add
// between a node and a literal, the literal should only be the
// 2nd operand. This preprocessing of the IR helps to simplify
// later passes.
class CanonicalizationPass : public FunctionPass {
 public:
  explicit CanonicalizationPass() : FunctionPass("canon", "Canonicalization") {}
  ~CanonicalizationPass() override {}

  xabsl::StatusOr<bool> RunOnFunction(Function* f, const PassOptions& options,
                                      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_CANONICALIZATION_PASS_H_
