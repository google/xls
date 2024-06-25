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

#include "xls/common/timeout_support.h"

#include <cstdlib>
#include <memory>
#include <thread>  // NOLINT

#include "absl/log/log.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

namespace xls {

namespace {
class Cleaner : public TimeoutCleaner {
 public:
  explicit Cleaner(absl::Duration duration)
      : duration_(duration), barrier_(2), thr_(Cleaner::StartThread, this) {
    // wait for the timeout thread to start.
    barrier_.Block();
  }

  ~Cleaner() override {
    notify_.Notify();
    thr_.join();
  }

 private:
  static void StartThread(Cleaner* thiz) {
    thiz->barrier_.Block();
    if (!thiz->notify_.WaitForNotificationWithTimeout(thiz->duration_)) {
      LOG(ERROR) << "Timeout of " << thiz->duration_ << " reached.";
      exit(1);
    }
  }
  absl::Duration duration_;
  absl::Barrier barrier_;
  absl::Notification notify_;
  std::thread thr_;
};
}  // namespace

std::unique_ptr<TimeoutCleaner> SetupTimeoutThread(absl::Duration duration) {
  return std::make_unique<Cleaner>(duration);
}

}  // namespace xls
