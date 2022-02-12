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

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// Pass which removes sends/receives that have literal false as their condition.
// Also removes the condition from sends/receives that have literal true as
// their condition.
class UselessIORemovalPass : public Pass {
 public:
  UselessIORemovalPass()
      : Pass("useless_io_remove", "Remove useless send/receive") {}
  ~UselessIORemovalPass() override {}

 protected:
  absl::StatusOr<bool> RunInternal(Package* p, const PassOptions& options,
                                   PassResults* results) const override;
};

}  // namespace xls

#endif
