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

#ifndef XLS_CONTRIB_ICE40_ICE40_IO_STRATEGY_H_
#define XLS_CONTRIB_ICE40_ICE40_IO_STRATEGY_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast.h"
#include "xls/contrib/ice40/io_strategy.h"
#include "xls/tools/verilog_include.h"

namespace xls {
namespace verilog {

class Ice40IoStrategy : public IOStrategy {
 public:
  explicit Ice40IoStrategy(VerilogFile* f);

  absl::Status AddTopLevelDependencies(LogicRef* clk, Reset reset,
                                       Module* m) override;

  absl::Status InstantiateIOBlocks(Input input, Output output,
                                   Module* m) override;

  absl::StatusOr<std::vector<VerilogInclude>> GetIncludes() override;

 private:
  // The files tick-included by the IO strategy.
  constexpr static const char* kIncludes[] = {
      "xls/contrib/ice40/uncore_rtl/ice40/uart_receiver.v",
      "xls/contrib/ice40/uncore_rtl/ice40/uart_transmitter.v"};

  VerilogFile* f_;

  // Signals we add as top level dependencies on the module.
  LogicRef* clk_ = nullptr;
  LogicRef* rst_n_ = nullptr;
  LogicRef* rx_in_ = nullptr;
  LogicRef* tx_out_ = nullptr;
  LogicRef* clear_to_send_ = nullptr;

  ParameterRef* clocks_per_baud_ = nullptr;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CONTRIB_ICE40_ICE40_IO_STRATEGY_H_
