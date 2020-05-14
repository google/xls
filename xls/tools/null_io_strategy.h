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

#ifndef THIRD_PARTY_XLS_TOOLS_NULL_IO_STRATEGY_H_
#define THIRD_PARTY_XLS_TOOLS_NULL_IO_STRATEGY_H_

#include "absl/status/status.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/statusor.h"
#include "xls/tools/io_strategy.h"

namespace xls {
namespace verilog {

// An IO strategy used for testing which transparently passes the byte-wise
// input and output interfaces up through the top-level module.
class NullIoStrategy : public IoStrategy {
 public:
  ~NullIoStrategy() override = default;

  absl::Status AddTopLevelDependencies(LogicRef1* clk, Reset reset,
                                       Module* m) override;

  absl::Status InstantiateIoBlocks(Input input, Output output,
                                   Module* m) override;

  xabsl::StatusOr<std::vector<VerilogInclude>> GetIncludes() override {
    return std::vector<VerilogInclude>();
  }

 private:
  // Top-level ports.
  LogicRef* byte_in_;
  LogicRef1* byte_in_ready_;
  LogicRef1* byte_in_valid_;

  LogicRef* byte_out_;
  LogicRef1* byte_out_ready_;
  LogicRef1* byte_out_valid_;
};

}  // namespace verilog
}  // namespace xls

#endif  // THIRD_PARTY_XLS_TOOLS_NULL_IO_STRATEGY_H_
