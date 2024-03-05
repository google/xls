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

#ifndef XLS_CODEGEN_REGISTER_CHAINING_ANALYSIS_H_
#define XLS_CODEGEN_REGISTER_CHAINING_ANALYSIS_H_

#include <deque>
#include <list>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"

namespace xls::verilog {

// Information about a register needed to construct chains.
struct RegisterData {
  // What register is being examined
  Register* reg;
  // The register read instruction
  RegisterRead* read;
  // The single stage where a read occurs
  Stage read_stage;
  // The register write instruction
  RegisterWrite* write;
  // The single stage where the write occurs
  Stage write_stage;

  template <typename H>
  friend H AbslHashValue(H h, const RegisterData& r) {
    return H::combine(std::move(h), r.reg, r.read, r.read_stage, r.write,
                      r.write_stage);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RegisterData& r) {
    absl::Format(&sink,
                 "%s (type: %s, read<stage: %d>: %s (id: %d), write<stage: "
                 "%d>: %s (id: %d))",
                 r.reg->name(), r.reg->type()->ToString(), r.read_stage,
                 r.read->GetName(), r.read->id(), r.write_stage,
                 r.write->GetName(), r.write->id());
  }
};

inline bool operator==(const RegisterData& l, const RegisterData& r) {
  return l.reg == r.reg && l.read == r.read && l.read_stage == r.read_stage &&
         l.write == r.write && l.write_stage == r.write_stage;
}

// A set of register chains that might be combined.
//
// A chain of registers is a sequence where a single value moves from one
// register to another down a series of stages.
//
// For chain analysis to work correctly a few things must be true of the IR and
// the input register data.
//
// First each register must only be submitted once and have a single read and
// write stage. Loopback registers in the IR might be physically written to
// multiple times but for chaining purposes we must use the earliest write only
// (since loopbacks terminate the chain).
//
// Second the load-enable input of all registers submitted must be used only as
// the stage activation bit. It may not be used as a 'poor-mans gate' or
// similar. Note that we must rely on the block-conversion to maintain this
// property (which it currently does).
//
// Third an activation must proceed monotonically down the pipeline. If in one
// cycle stages A & B are activated (and there are at least max(A, B) + 1
// stages) then the next cycle the activated stages must be a superset of one of
// {A, B}, {A + 1, B}, {A, B + 1}, or {A + 1, B + 1}, This is because the
// register chains track the motion of a single value though a set of registers
// so if activation is skipping around this analysis does not function.
//
// In terms of load enables this can be stated as:
//
// Suppose you have the following data flow graph:
//
//   R_0 -> R_1 -> {S side effecting operations}
//
// R_0 and R_1 are registers with load enables le_0 and le_1 The output of R_0
// directly feeds the input of R_1. S is a set of side-effecting operations. S
// can include registers (including R_0). S only includes operations
// combinationally connected to R_0 (no intervening registers), or alternatively
// S are the effects in the next downstream stage.
//
// You can collapse R_0 into R_1 under the following conditions:
//   - an assertion of le_1 occurs strictly after an assertion of le_0.
//   - the side-effecting operations S are activated strictly after an assertion
//     of le_1
//   - the side-effecting operations S must be activated at or before the next
//     assertion of le_0
class RegisterChains {
 public:
  // Add the given register into a chain if possible or create a new one.
  //
  // Once it is in a chain the chain set is reduced to produce as few chains as
  // possible.
  void InsertAndReduce(const RegisterData& data);

  // Get the mutex chains which exist within the mutually-exclusive groups
  // marked by 'groups'.
  absl::StatusOr<std::vector<std::vector<RegisterData>>>
  SplitBetweenMutexRegions(const ConcurrentStageGroups& groups,
                           const CodegenPassOptions& options) const;

  // The current in-progress chains. Use SplitBetweenMutexRegions to get the
  // final mutex-chains.
  const std::list<std::deque<RegisterData>>& chains() const { return chains_; }

 private:
  std::list<std::deque<RegisterData>> chains_;
};
}  // namespace xls::verilog

#endif  // XLS_CODEGEN_REGISTER_CHAINING_ANALYSIS_H_
