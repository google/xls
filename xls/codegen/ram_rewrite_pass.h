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

#include <cstdint>

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
// ready/valid ports will be replaced by internal signals to/from a buffer added
// to catch the output of the RAM. Note: this pass should generally be used in
// conjunction with a scheduling constraint to ensure that the receive on the
// response channel comes a cycle after the send on the request channel.
//
// A RAM can't be stalled, so its read responses must all be caught, even if
// their consumer isn't ready yet; the request side is therefore always ready,
// and the response buffer must be able to hold every read response that can be
// outstanding (issued but not yet consumed) at once. That number depends on
// when the pipeline issues reads, so it's provided by the caller as
// `max_outstanding_reads`:
//
//  - Codegen 1.0 only issues a read in a cycle where the requesting stage
//    advances, so at most 1 response can be outstanding.
//  - Codegen 1.5 can complete a stage's sends (each at most once per
//    activation) while the stage is stalled, so a read can be issued while the
//    previous activation's response is still waiting for its consumer; at most
//    2 responses can be outstanding.
class RamRewritePass : public CodegenPass {
 public:
  explicit RamRewritePass(int64_t max_outstanding_reads)
      : CodegenPass("ram_channel_rewrite", "Rewrite channels to RAMs"),
        max_outstanding_reads_(max_outstanding_reads) {}
  ~RamRewritePass() override = default;

  absl::StatusOr<bool> RunInternal(Package* package,
                                   const CodegenPassOptions& options,
                                   PassResults* results,
                                   CodegenContext& context) const final;

 private:
  int64_t max_outstanding_reads_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_RAM_REWRITE_PASS_H_
