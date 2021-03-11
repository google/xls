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

#ifndef XLS_IR_IR_INTERPRETER_STATS_H_
#define XLS_IR_IR_INTERPRETER_STATS_H_

#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"

namespace xls {

// Note: as of now this is more of a "performance counter" dumb-struct sort of
// class, where the determination of when/where to note things is inline in the
// IR interpreter itself.
class InterpreterStats {
 public:
  void NoteShllAmountForBitCount(int64_t amount, int64_t bit_count) {
    absl::MutexLock lock(&mutex_);
    all_shlls_ += 1;
    overlarge_shlls_ += amount >= bit_count;
    zero_shlls_ += amount == 0;
  }

  // Notes the bits result for a given node (as determined by the interpreter)
  // -- the values that have consistent bits are recorded via the "Meet"
  // operator.
  void NoteNodeBits(std::string node_string, const Bits& bits) {
    absl::MutexLock lock(&mutex_);
    auto& lattice = value_profile_[node_string];
    Meet(bits, &lattice);
  }

  // Returns a multi-line report string suitable for, e.g. XLS_LOG_LINES'ing.
  std::string ToReport() const;

 private:
  int64_t in_range_shlls() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return all_shlls_ - overlarge_shlls_ - zero_shlls_;
  }

  // Meets the observed "bits" value against the seen-so-far "lattice" of values
  // -- e.g. if the seen-so-far value is 0 and "bits" contains a 1 in that bit
  // position, the value will go to "bottom" (unknown = X).
  //
  // A nullopt in the lattice means no value has yet been observed for the given
  // node.
  void Meet(const Bits& bits, absl::optional<TernaryVector>* lattice) {
    if (lattice->has_value()) {
      XLS_CHECK_EQ(bits.bit_count(), lattice->value().size());
      *lattice = ternary_ops::Equals(ternary_ops::BitsToTernary(bits),
                                     lattice->value());
    } else {
      *lattice = ternary_ops::BitsToTernary(bits);
    }
  }

  // Returns a string that represents the nodes with consistent bit values.
  std::string ToNodeReport() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;

  // When absl::nullopt, the value is "top" in the lattice (i.e. no info is
  // available).
  //
  // When the ternary value is present, kUnknown is "bottom" in the lattice
  // (conflicting info).
  absl::flat_hash_map<std::string, absl::optional<TernaryVector>> value_profile_
      ABSL_GUARDED_BY(mutex_);
  int64_t overlarge_shlls_ ABSL_GUARDED_BY(mutex_) = 0;
  int64_t zero_shlls_ ABSL_GUARDED_BY(mutex_) = 0;
  int64_t all_shlls_ ABSL_GUARDED_BY(mutex_) = 0;
};

}  // namespace xls

#endif  // XLS_IR_IR_INTERPRETER_STATS_H_
