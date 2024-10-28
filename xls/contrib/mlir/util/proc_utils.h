// Copyright 2024 The XLS Authors
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

#ifndef XLS_CONTRIB_MLIR_UTIL_PROC_UTILS_H_
#define XLS_CONTRIB_MLIR_UTIL_PROC_UTILS_H_

#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir::xls {

// Converts a ForOp to a call (send/recv pair) to a SprocOp.
//
// Two new SprocOps are created:
//   - A "body" SprocOp that contains the body of the ForOp.
//   - A "controller" SprocOp that contains a state machine that calls the body
//     SprocOp.
//
// The ForOp is replaced with a call to the "controller" SprocOp.
//
// The ForOp's immediate parent must be an SprocOp.
//
// Returns failure if the ForOp's trip count cannot be determined.
//
// IMPLEMENTATION DETAILS
// ======================
//
// Inside the ForOp's original sproc parent, we create channels and spawn a
// "controller" SprocOp. The ForOp's original site becomes:
//
//   %tkn = xls.after_all()
//   %tkn2 = xls.ssend(%tkn, %for_operands..., %captured_invariants...)
//   _, %result = xls.sblocking_receive(%tkn2)
//
// This pseudocode shows one ssend and one sblocking_receive, but in practice
// there is one send per for each of [operands..., invariants...] and one
// receive per for each of [results...].
//
// The controller SprocOp is a state machine that itself spawns a "body"
// SprocOp.
//
// Simplified: suppose the original loop looked like:
//
// %results = scf.for (%operands, %init_args)
//   ^bb(%body_iter_args, %indvar) {
//     ....
//     yield %body_results
// }
//
// The state machine we generate for the _controller_ is:
//
// state <- 0
// while True:
//   if state == 0:
//      iter_args <- recv(init_args_chans)
//   else:
//      iter_args <- recv(body_results_chans)
//  if state == kTripCount:
//     send(iter_args to results_chans)
//     state <- 0
//  else:
//      send(iter_args to body_iter_args_chans)
//      send(state to body_indvar_chan)
//      state <- state+1
//
// And the body SprocOp is simply the ForOp's body region, but with arguments
// replaced by sblocking_receives and the terminator replaced by a ssend of each
// result. The body receives from body_iter_args_chans, body_indvar_chan and
// sends to body_results_chans.
//
// The controller receives from operands_chans, body_results_chans and sends to
// body_iter_args_chans, body_indvar_chan and results_chans.
//
// The "invariants" mentioned above are values that are used within the body but
// defined outside the body region. They are made explicit, captured at the
// ForOp's site and plumbed through to the body. Conceptually we can simply
// make the ForOp isolatedFromAbove, capturing the invariants as iter_args, but
// we plumb them through explicitly instead to allow for optimizations.
//
// The SymbolTable is updated.
LogicalResult convertForOpToSprocCall(scf::ForOp forOp,
                                      SymbolTable& symbolTable);

// Registers the test pass for extracting as a top level module.
namespace test {
void registerTestConvertForOpToSprocCallPass();
}

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_UTIL_PROC_UTILS_H_
