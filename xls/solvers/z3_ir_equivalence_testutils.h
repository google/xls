// Copyright 2023 The XLS Authors
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

#ifndef XLS_SOLVERS_Z3_IR_EQUIVALENCE_TESTUTILS_H_
#define XLS_SOLVERS_Z3_IR_EQUIVALENCE_TESTUTILS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/common/source_location.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls::solvers::z3 {

class ScopedVerifyEquivalence {
 public:
  explicit ScopedVerifyEquivalence(
      Function* f, absl::Duration timeout = absl::InfiniteDuration(),
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  ~ScopedVerifyEquivalence();

 private:
  Function* const f_;
  absl::Duration timeout_;
  const xabsl::SourceLocation loc_;

  std::unique_ptr<Package> clone_p_;
  Function* original_f_;
};

// A helper to verify that proc behavior (as measured by the sent-values per
// cycle) has not changed.
//
// Note this has several limitations that users must be aware of. First token
// dependency order is not checked in any fashion, modifications which break
// dependency restrictions will not be detected. Second, only a single send on
// each channel may be present in the IR, conditional send is not supported.
// Third all incoming channels are ready and all outgoing channels are valid at
// all times. Fourth, the proc may not use the 'invoke' instruction or any
// others that the z3 equivalence checker does not support.
//
// This checker does not perform temporal reasoning about the continued
// evolution of the channel values or proc state and cannot be used as a general
// equivalence proof. This is meant for test use and basic correctness checking
// only.
//
// The user should also be sure to pick an appropriate activation count since
// this is used internally. If some interesting state will not come around until
// after 'activation_count' activations it will not be explored.
class ScopedVerifyProcEquivalence {
 public:
  ScopedVerifyProcEquivalence(ScopedVerifyProcEquivalence&&) = default;
  ScopedVerifyProcEquivalence(const ScopedVerifyProcEquivalence&) = delete;
  ScopedVerifyProcEquivalence& operator=(ScopedVerifyProcEquivalence&&) =
      delete;
  ScopedVerifyProcEquivalence& operator=(const ScopedVerifyProcEquivalence&) =
      delete;

  ScopedVerifyProcEquivalence(
      Proc* p, int64_t activation_count, bool include_state,
      absl::Duration timeout = absl::InfiniteDuration(),
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  ~ScopedVerifyProcEquivalence();

 private:
  void RunProcVerification();

  Proc* const p_;
  int64_t activation_count_;
  bool include_state_;
  absl::Duration timeout_;
  const xabsl::SourceLocation loc_;

  std::unique_ptr<Package> clone_package_;
  Proc* original_p_;
};

// A helper to verify that block behavior (as measured by the port-values per
// cycle) has not changed.
//
// Note this has several limitations that users must be aware of. First, by
// default the simulation has very limited awareness of 'ready/valid/data'
// signaling and considers all data to be unconditionally consumed. If
// 'SetZeroInvalidChannelData' is called with the default argument value of
// true, then any invalid channel data is forced to zero if the corresponding
// valid bit is not set. This relies on metadata created during block conversion
// to be present and correct however.
//
// This checker does not perform temporal reasoning about the continued
// evolution of the port values or register state and cannot be used as a
// general equivalence proof. This is meant for test use and basic correctness
// checking only.
//
// The user should also be sure to pick an appropriate tick count since this is
// used internally. If some interesting state cannot come around until after
// 'tick_count' ticks it will not be explored. Note that the time it takes for
// z3 to prove facts about the block can increase rapidly as the tick count
// increases.
//
// If 'SetIncludeRegState' is called then the register state at each tick will
// also be compared. The registers are correlated based on the order they appear
// in the block.
//
// NB Internally this works by transforming the block into a function with
// each tick unrolled.
class ScopedVerifyBlockEquivalence {
 public:
  ScopedVerifyBlockEquivalence(ScopedVerifyBlockEquivalence&&) = default;
  ScopedVerifyBlockEquivalence(const ScopedVerifyBlockEquivalence&) = delete;
  ScopedVerifyBlockEquivalence& operator=(ScopedVerifyBlockEquivalence&&) =
      delete;
  ScopedVerifyBlockEquivalence& operator=(const ScopedVerifyBlockEquivalence&) =
      delete;

  ScopedVerifyBlockEquivalence(
      Block* b, int64_t tick_count, bool zero_invalid_channel_data = false,
      bool include_reg_state = false,
      absl::Duration timeout = absl::InfiniteDuration(),
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  ~ScopedVerifyBlockEquivalence();

  ScopedVerifyBlockEquivalence& SetZeroInvalidChannelData(
      bool zero_invalid_channel_data = true) {
    zero_invalid_channel_data_ = zero_invalid_channel_data;
    return *this;
  }

  ScopedVerifyBlockEquivalence& SetIncludeRegState(
      bool include_reg_state = true) {
    include_reg_state_ = include_reg_state;
    return *this;
  }

 private:
  void RunBlockVerification();

  Block* const b_;
  int64_t tick_count_;
  bool zero_invalid_channel_data_;
  bool include_reg_state_;
  absl::Duration timeout_;
  const xabsl::SourceLocation loc_;

  std::unique_ptr<Package> clone_package_;
  Block* original_b_;
};

// Print a function with each node annotated with its value in the given
// counterexample.
absl::StatusOr<std::string> DumpWithNodeValues(Function* func,
                                               const ProvenFalse& fail);

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_IR_EQUIVALENCE_TESTUTILS_H_
