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

#include "xls/simulation/generic/iirq_stub.h"

#include <functional>

#include "absl/status/status.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic {

IIRQStub::IIRQStub() : status_(false), policy_([] { return true; }) {}

bool IIRQStub::GetIRQ() {
  XLS_LOG(INFO) << "IIRQStub::GetIRQ()";
  return this->status_;
}

absl::Status IIRQStub::UpdateIRQ() {
  XLS_LOG(INFO) << "IIRQStub::UpdateIRQ()";
  this->status_ = this->policy_();
  return absl::OkStatus();
}

void IIRQStub::SetPolicy(std::function<bool(void)> policy) {
  this->policy_ = policy;
}

}  // namespace xls::simulation::generic
