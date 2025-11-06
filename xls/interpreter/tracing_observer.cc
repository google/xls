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

#include "xls/interpreter/tracing_observer.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"

namespace xls {

void TracingObserver::NodeEvaluated(Node* node, const Value& result) {
  absl::Status status = recorder_.RecordNodeValue(node, result);
  if (!status.ok()) {
    // Log error but don't stop evaluation.
    LOG(ERROR) << "Error recording node value: " << status;
  }
}

void TracingObserver::Tick() { recorder_.Tick(); }

void ScopedTracingObserver::NodeEvaluated(Node* node, const Value& result) {
  observer_.NodeEvaluated(node, result);
}

void ScopedTracingObserver::Tick() { observer_.Tick(); }

ScopedTracingObserver::~ScopedTracingObserver() {
  if (!writer_->Close()) {
    LOG(ERROR) << "Error writing trace";
  }
}

}  // namespace xls
