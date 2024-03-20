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

#include "xls/simulation/generic/iactive_stub.h"

#include "absl/status/status.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic {

IActiveStub::IActiveStub() : cnt_(0) {}

absl::Status IActiveStub::Update() {
  XLS_LOG(INFO) << "IActiveStub tick: " << this->cnt_;

  ++this->cnt_;
  return absl::OkStatus();
}

int IActiveStub::getCnt() { return this->cnt_; }

void IActiveStub::setCnt(int cnt) { this->cnt_ = cnt; }

}  // namespace xls::simulation::generic
