// Copyright 2025 The XLS Authors
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

#ifndef XLS_CONTRIB_XLSCC_TRACKED_BVALUE_H_
#define XLS_CONTRIB_XLSCC_TRACKED_BVALUE_H_

#include <map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"

#define DEBUG_SAVE_BACKTRACES 0

namespace xlscc {

// The TrackedBValue class is a wrapper for xls::BValue that enables getting an
// ordered list of every BValue currently instantiated in XLS[cc].
//
// This is deterministic because XLS[cc] generates deterministic IR, and
// xls::BValues are created in IR generation.
//
// Continuations enable splitting a function into slices, which may be placed
// into different proc activations. They expose every "live" value in XLS[cc],
// which could be referenced after the slice point, at the time the split is
// created.
//
// The reason it is insufficient to generate continuations for each variable
// is that values are kept by XLS[cc] in various places scattered around the
// code for later use. For example, in an if-else statement, the condition for
// the if is kept on the stack to be later negated for the else.
//
// Not thread safe
class TrackedBValue {
 public:
  class Lock;
  friend class Lock;

  TrackedBValue() : sequence_number_(sNextSequenceNumber++) {
    // Nothing to record
  }
  TrackedBValue(const TrackedBValue& bval)
      : bval_(bval.bval_), sequence_number_(sNextSequenceNumber++) {
    record();
  }
  TrackedBValue(const xls::BValue& native_bval)
      : bval_(native_bval), sequence_number_(sNextSequenceNumber++) {
    record();
  }
  TrackedBValue(xls::Node* node, xls::BuilderBase* builder)
      : bval_(node, builder), sequence_number_(sNextSequenceNumber++) {
    record();
  }
  TrackedBValue(TrackedBValue&& o) : sequence_number_(sNextSequenceNumber++) {
    o.unrecord();
    bval_ = std::move(o.bval_);
    // Destructor is still called on the source object
    o.bval_ = xls::BValue();
    record();
  }
  ~TrackedBValue() { unrecord(); }
  TrackedBValue& operator=(const TrackedBValue& bval) {
    unrecord();
    bval_ = bval.bval_;
    record();
    return *this;
  }
  operator xls::BValue() const { return bval_; }
  bool valid() const { return bval_.valid(); }
  xls::Node* node() const { return bval_.node(); }
  std::string ToString() const { return bval_.ToString(); }
  xls::Type* GetType() const { return bval_.GetType(); }
  int64_t BitCountOrDie() const { return bval_.BitCountOrDie(); }
  const xls::SourceInfo& loc() const { return bval_.loc(); }

  xls::BuilderBase* builder() const { return bval_.builder(); }

  int64_t sequence_number() const {
    CHECK_GT(sequence_number_, 0);
    return sequence_number_;
  }

  class Lock {
    friend class TrackedBValue;

   public:
    Lock(const Lock& o) = delete;
    Lock(Lock&& o);
    ~Lock();

    void UnlockEarly();

   private:
    Lock();
    bool locked_ = false;
  };

  static std::tuple<Lock, std::vector<TrackedBValue*>> OrderedBValuesForBuilder(
      xls::BuilderBase* builder);

  static void RegisterBuilder(xls::BuilderBase* builder);
  static void UnregisterBuilder(xls::BuilderBase* builder);

 private:
  void record();
  void unrecord();

  xls::BValue bval_;
  int64_t sequence_number_ = -1;

#if DEBUG_SAVE_BACKTRACES
  std::string debug_backtrace_;
#endif  // DEBUG_SAVE_BACKTRACES

  static int64_t sNextSequenceNumber;

  static bool sLocked;

  static absl::flat_hash_map<xls::BuilderBase*,
                             absl::flat_hash_set<TrackedBValue*>>
      sTrackedBValuesByBuilder;
};

class TrackedFunctionBuilder {
 public:
  TrackedFunctionBuilder(std::string_view name, xls::Package* package,
                         bool should_verify = true)
      : builder_(name, package, should_verify) {
    TrackedBValue::RegisterBuilder(&builder_);
  }
  ~TrackedFunctionBuilder() { TrackedBValue::UnregisterBuilder(&builder_); }
  xls::FunctionBuilder* builder() { return &builder_; }

 private:
  xls::FunctionBuilder builder_;
};

class TrackedProcBuilder {
 public:
  TrackedProcBuilder(std::string_view name, xls::Package* package,
                     bool should_verify = true)
      : builder_(name, package, should_verify) {
    TrackedBValue::RegisterBuilder(&builder_);
  }
  ~TrackedProcBuilder() { TrackedBValue::UnregisterBuilder(&builder_); }
  xls::ProcBuilder* builder() { return &builder_; }

 private:
  xls::ProcBuilder builder_;
};

typedef xls::BValue NATIVE_BVAL;

std::vector<NATIVE_BVAL> ToNativeBValues(
    const std::vector<TrackedBValue>& bvals);

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_TRACKED_BVALUE_H_
