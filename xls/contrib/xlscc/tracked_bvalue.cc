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
#include "xls/ir/function_builder.h"

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

TrackedBValue::Lock::~Lock() {
  if (locked_) {
    CHECK(TrackedBValue::sLocked);
    TrackedBValue::sLocked = false;
  }
  locked_ = false;
}

TrackedBValue::Lock::Lock() : locked_(true) { CHECK(!TrackedBValue::sLocked); }

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

void TrackedBValue::record() {
  CHECK(!sLocked);
  if (!bval_.valid()) {
    return;
  }
  CHECK(!sTrackedBValuesByBuilder.contains(bval_.builder()) ||
        !sTrackedBValuesByBuilder.at(bval_.builder()).contains(this));
  sTrackedBValuesByBuilder[bval_.builder()].insert(this);
}

void TrackedBValue::unrecord() {
  CHECK(!sLocked);
  if (!bval_.valid()) {
    return;
  }
  CHECK(sTrackedBValuesByBuilder.contains(bval_.builder()) &&
        sTrackedBValuesByBuilder.at(bval_.builder()).contains(this));
  sTrackedBValuesByBuilder.at(bval_.builder()).erase(this);
  if (sTrackedBValuesByBuilder.at(bval_.builder()).empty()) {
    sTrackedBValuesByBuilder.erase(bval_.builder());
  }
}

absl::flat_hash_map<xls::BuilderBase*, absl::flat_hash_set<TrackedBValue*>>
    TrackedBValue::sTrackedBValuesByBuilder;

int64_t TrackedBValue::sNextSequenceNumber = 1;

bool TrackedBValue::sLocked = false;

}  // namespace xlscc
