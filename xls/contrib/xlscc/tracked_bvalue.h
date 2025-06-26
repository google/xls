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

#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/source_location.h"

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
  // NOLINTNEXTLINE(google-explicit-constructor)
  TrackedBValue(const xls::BValue& native_bval)
      : bval_(native_bval), sequence_number_(sNextSequenceNumber++) {
    record();
  }
  TrackedBValue(xls::Node* node, xls::BuilderBase* builder)
      : bval_(node, builder), sequence_number_(sNextSequenceNumber++) {
    record();
  }
  // Keeping the same sequence number is safe here, and it allows TrackedBValues
  // to be stored in containers such as maps without changing sequence numbers.
  TrackedBValue(TrackedBValue&& o) {
    o.unrecord();
    bval_ = std::move(o.bval_);
    sequence_number_ = o.sequence_number_;
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
  void destroy() {
    unrecord();
    bval_ = xls::BValue();
  }
  operator xls::BValue() const {  // NOLINT(google-explicit-constructor)
    return bval_;
  }
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

#if DEBUG_SAVE_BACKTRACES
  const std::string& debug_backtrace() const { return debug_backtrace_; }
#endif  // DEBUG_SAVE_BACKTRACES

  class Lock {
    friend class TrackedBValue;

   public:
    Lock(const Lock& o) = delete;
    Lock(Lock&& o);
    ~Lock();

   private:
    Lock();
    bool locked_ = false;
  };

  static std::tuple<Lock, std::vector<TrackedBValue*>> OrderedBValuesForBuilder(
      xls::BuilderBase* builder);
  static std::vector<TrackedBValue*> OrderBValues(
      const absl::flat_hash_set<TrackedBValue*>& bvals_unordered);
  static std::vector<const TrackedBValue*> OrderBValues(
      const absl::flat_hash_set<const TrackedBValue*>& bvals_unordered);

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

// This class exists to wrap maps containing TrackedBValues such that when they
// are copied the values are inserted in a deterministic order. This maintains
// the deterministic sequence numbers for the TrackedBValues contained within.
template <typename K, typename V, typename MapType, typename OrderValuesFunc>
class DeterministicMapBase : public MapType {
 public:
  DeterministicMapBase() = default;
  DeterministicMapBase(DeterministicMapBase&&) = default;

  DeterministicMapBase(const DeterministicMapBase& o) {
    CopyDeterministicallyFrom(o);
  }

  DeterministicMapBase& operator=(const DeterministicMapBase& o) {
    CopyDeterministicallyFrom(o);
    return *this;
  }

 private:
  void CopyDeterministicallyFrom(const DeterministicMapBase& o) {
    absl::flat_hash_map<const V*, K> key_by_bval;
    absl::flat_hash_set<const V*> bvals;

    for (auto& [key, bval] : o) {
      key_by_bval[&bval] = key;
      bvals.insert(&bval);
    }

    std::vector<const V*> ref_ordered = OrderValuesFunc()(bvals);

    absl::flat_hash_map<K, V>::clear();
    for (const V* bval : ref_ordered) {
      (*this)[key_by_bval.at(bval)] = *bval;
    }
  }
};

class OrderTrackedBValuesFunc {
 public:
  std::vector<const TrackedBValue*> operator()(
      const absl::flat_hash_set<const TrackedBValue*>& bvals_unordered);
};

template <typename K>
using TrackedBValueMap =
    DeterministicMapBase<K, TrackedBValue,
                         absl::flat_hash_map<K, TrackedBValue>,
                         OrderTrackedBValuesFunc>;

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_TRACKED_BVALUE_H_
