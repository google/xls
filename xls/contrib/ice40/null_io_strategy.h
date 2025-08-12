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

#ifndef XLS_CONTRIB_ICE40_NULL_IO_STRATEGY_H_
#define XLS_CONTRIB_ICE40_NULL_IO_STRATEGY_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast/vast.h"
#include "xls/contrib/ice40/io_strategy.h"
#include "xls/simulation/verilog_include.h"

namespace xls {
namespace verilog {

// An IO strategy used for testing which transparently passes the byte-wise
// input and output interfaces up through the top-level module.
class NullIOStrategy final : public IOStrategy {
 public:
  ~NullIOStrategy() final = default;

  absl::Status AddTopLevelDependencies(LogicRef* clk, Reset reset,
                                       Module* m) final;

  absl::Status InstantiateIOBlocks(Input input, Output output,
                                   Module* m) final;

  absl::StatusOr<std::vector<VerilogInclude>> GetIncludes() final {
    return std::vector<VerilogInclude>();
  }

 private:
  // Top-level ports.
  LogicRef* byte_in_;
  LogicRef* byte_in_ready_;
  LogicRef* byte_in_valid_;

  LogicRef* byte_out_;
  LogicRef* byte_out_ready_;
  LogicRef* byte_out_valid_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CONTRIB_ICE40_NULL_IO_STRATEGY_H_
