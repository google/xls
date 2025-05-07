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

#include "xls/contrib/xlscc/tracked_bvalue.h"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xls/ir/function_builder.h"

#if DEBUG_SAVE_BACKTRACES
#include "stacktrace.h"
#endif

namespace xlscc {

std::vector<NATIVE_BVAL> ToNativeBValues(
    const std::vector<TrackedBValue>& bvals) {
  std::vector<NATIVE_BVAL> native_bvals;
  native_bvals.reserve(bvals.size());
  for (const TrackedBValue& bval : bvals) {
    native_bvals.push_back(bval);
  }
  return native_bvals;
}

TrackedBValue::Lock::Lock(Lock&& o) {
  locked_ = o.locked_;
  o.locked_ = false;
}

void TrackedBValue::Lock::UnlockEarly() {
  if (locked_) {
    CHECK(TrackedBValue::sLocked);
    TrackedBValue::sLocked = false;
  }
  locked_ = false;
}

TrackedBValue::Lock::~Lock() { UnlockEarly(); }

TrackedBValue::Lock::Lock() : locked_(true) {
  CHECK(!TrackedBValue::sLocked);
  TrackedBValue::sLocked = true;
}

std::tuple<TrackedBValue::Lock, std::vector<TrackedBValue*>>
TrackedBValue::OrderedBValuesForBuilder(xls::BuilderBase* builder) {
  if (!sTrackedBValuesByBuilder.contains(builder)) {
    return std::tuple<TrackedBValue::Lock, std::vector<TrackedBValue*>>{
        TrackedBValue::Lock(), std::vector<TrackedBValue*>()};
  }
  absl::flat_hash_set<TrackedBValue*> bvals_unordered =
      sTrackedBValuesByBuilder.at(builder);
  std::vector<TrackedBValue*> ret;
  ret.reserve(bvals_unordered.size());
  ret.insert(ret.end(), bvals_unordered.begin(), bvals_unordered.end());
  std::sort(ret.begin(), ret.end(), [](TrackedBValue* a, TrackedBValue* b) {
    return a->sequence_number() < b->sequence_number();
  });

  return std::tuple<TrackedBValue::Lock, std::vector<TrackedBValue*>>{
      TrackedBValue::Lock(), ret};
}
void TrackedBValue::RegisterBuilder(xls::BuilderBase* builder) {
  CHECK(!sLocked);
  CHECK(!sTrackedBValuesByBuilder.contains(builder));
  sTrackedBValuesByBuilder[builder] = {};
}

void TrackedBValue::UnregisterBuilder(xls::BuilderBase* builder) {
  CHECK(!sLocked);
  CHECK(sTrackedBValuesByBuilder.contains(builder));

  const absl::flat_hash_set<TrackedBValue*>& bvals =
      sTrackedBValuesByBuilder.at(builder);
  if (!bvals.empty()) {
    LOG(WARNING) << absl::StrFormat("bvals left over for builder %s:\n",
                                    builder->name().c_str());
    for (TrackedBValue* bval : bvals) {
      LOG(WARNING) << absl::StrFormat("--- (builder %p) %p: %s\n", builder,
                                      bval->node(), bval->node()->ToString());

#if DEBUG_SAVE_BACKTRACES
      LOG(WARNING) << absl::StrFormat("TRACE:\n%s\n-----\n",
                                      bval->debug_backtrace_);
#endif  // DEBUG_SAVE_BACKTRACES
    }
  }
  CHECK(sTrackedBValuesByBuilder.at(builder).empty());
  sTrackedBValuesByBuilder.erase(builder);
}

void TrackedBValue::record() {
  CHECK(!sLocked);
  if (!bval_.valid()) {
    return;
  }
  CHECK(sTrackedBValuesByBuilder.contains(bval_.builder()));
  CHECK(!sTrackedBValuesByBuilder.at(bval_.builder()).contains(this));
  sTrackedBValuesByBuilder[bval_.builder()].insert(this);
#if DEBUG_SAVE_BACKTRACES
  debug_backtrace_ =
      getstacktrace(/*max_depth=*/50);
#endif  // DEBUG_SAVE_BACKTRACES
}

void TrackedBValue::unrecord() {
  CHECK(!sLocked);
  if (!bval_.valid()) {
    return;
  }
  CHECK(sTrackedBValuesByBuilder.contains(bval_.builder()));
  CHECK(sTrackedBValuesByBuilder.at(bval_.builder()).contains(this));
  sTrackedBValuesByBuilder.at(bval_.builder()).erase(this);
}

absl::flat_hash_map<xls::BuilderBase*, absl::flat_hash_set<TrackedBValue*>>
    TrackedBValue::sTrackedBValuesByBuilder;

int64_t TrackedBValue::sNextSequenceNumber = 1;

bool TrackedBValue::sLocked = false;

}  // namespace xlscc
