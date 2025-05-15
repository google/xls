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

#ifndef XLS_CODEGEN_RAM_REWRITE_PASS_H_
#define XLS_CODEGEN_RAM_REWRITE_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

// Rewrites selected channels to work with RAMs.
//
// Request channels must be a 4-tuple, where the entries are (addr, wr_data, we,
// re). Response channels must be a 1-tuple where the entry is (rd_data). The
// selected channels will be rewritten by this pass to support interacting with
// external RAMs; in particular, the request data port will expanded into 4
// ports for each element of the tuple (addr, wr_data, we, re). Furthermore,
// ready/valid ports will be replaced by internal signals to/from a skid buffer
// added to catch the output of the RAM. Note: this pass should generally be
// used in conjunction with a scheduling constraint to ensure that the receive
// on the response channel comes a cycle after the send on the request channel.
class RamRewritePass : public CodegenPass {
 public:
  RamRewritePass()
      : CodegenPass("ram_channel_rewrite", "Rewrite channels to RAMs") {}
  ~RamRewritePass() override = default;

  absl::StatusOr<bool> RunInternal(Package* package,
                                   const CodegenPassOptions& options,
                                   PassResults* results,
                                   CodegenContext& context) const final;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_RAM_REWRITE_PASS_H_
