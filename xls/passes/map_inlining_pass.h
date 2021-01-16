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

#ifndef XLS_PASSES_MAP_INLINING_PASS_H_
#define XLS_PASSES_MAP_INLINING_PASS_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/passes/passes.h"

namespace xls {

// A pass to convert map nodes to in-line Invoke nodes. We don't directly lower
// maps to Verilog.
class MapInliningPass : public FunctionBasePass {
 public:
  MapInliningPass();

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* function, const PassOptions& options,
      PassResults* results) const override;

  // Replaces a single Map node with a CountedFor operation.
  absl::Status ReplaceMap(Map* map) const;
};

}  // namespace xls

#endif  // XLS_PASSES_MAP_INLINING_PASS_H_
