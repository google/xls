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

#include "xls/tools/null_io_strategy.h"

namespace xls {
namespace verilog {

absl::Status NullIoStrategy::AddTopLevelDependencies(LogicRef1* clk,
                                                     Reset reset, Module* m) {
  byte_in_ = m->AddPort(Direction::kInput, "byte_in", 8);
  byte_in_ready_ = m->AddOutput("byte_in_ready");
  byte_in_valid_ = m->AddInput("byte_in_valid");

  byte_out_ = m->AddPort(Direction::kOutput, "byte_out", 8);
  byte_out_ready_ = m->AddInput("byte_out_ready");
  byte_out_valid_ = m->AddOutput("byte_out_valid");

  return absl::OkStatus();
}

absl::Status NullIoStrategy::InstantiateIoBlocks(Input input, Output output,
                                                 Module* m) {
  m->Add<ContinuousAssignment>(input.rx_byte, byte_in_);
  m->Add<ContinuousAssignment>(byte_in_ready_, input.rx_byte_done);
  m->Add<ContinuousAssignment>(input.rx_byte_valid, byte_in_valid_);

  m->Add<ContinuousAssignment>(byte_out_, output.tx_byte);
  m->Add<ContinuousAssignment>(output.tx_byte_ready, byte_out_ready_);
  m->Add<ContinuousAssignment>(byte_out_valid_, output.tx_byte_valid);

  return absl::OkStatus();
}

}  // namespace verilog
}  // namespace xls
