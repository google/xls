// Copyright 2022 The XLS Authors
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

#ifndef XLS_PASSES_USELESS_IO_REMOVAL_PASS_H_
#define XLS_PASSES_USELESS_IO_REMOVAL_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which removes sends/receives that have literal false as their condition.
// Also removes the condition from sends/receives that have literal true as
// their condition.
class UselessIORemovalPass : public OptimizationPass {
 public:
  static constexpr std::string_view kName = "useless_io_remove";
  UselessIORemovalPass()
      : OptimizationPass(kName, "Remove useless send/receive") {}
  ~UselessIORemovalPass() override = default;

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results) const override;
};

}  // namespace xls

#endif
