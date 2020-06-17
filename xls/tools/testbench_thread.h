// Copyright 2020 Google LLC
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

#ifndef XLS_TOOLS_TESTBENCH_THREAD_H_
#define XLS_TOOLS_TESTBENCH_THREAD_H_

#include <functional>
#include <thread>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/llvm_ir_jit.h"
#include "xls/ir/package.h"

namespace xls {

// TestbenchThread handles the work of _actually_ running tests.
// It simply iterates over its given range of the index space and calls the
// expected/actual calculators.
template <typename InputT, typename ResultT>
class TestbenchThread {
 public:
  // All specified functions must be thread-safe.
  //  - wake_parent_mutex: A mutex that protects:
  //  - wake_parent: A condvar to kick the parent when this thread has finished.
  //  - max_failures: The number of failures that will cause us to bail out.
  //                  If 0, then there will be no limit.
  //  - index_to_input: A function that can convert an index to an input to the
  //                    calculation routines.
  //  - generate_expected: Given an input, generates the "expected" value.
  //  - generate_actual: Given an input, generates a value from the module
  //                     under test.
  TestbenchThread(const std::string& ir_text, const std::string& entry_fn,
                  absl::Mutex* wake_parent_mutex, absl::CondVar* wake_parent,
                  uint64 start_index, uint64 end_index, uint64 max_failures,
                  std::function<InputT(uint64)> index_to_input,
                  std::function<ResultT(InputT)> generate_expected,
                  std::function<ResultT(LlvmIrJit*, absl::Span<uint8>, InputT)>
                      generate_actual,
                  std::function<bool(ResultT, ResultT)> compare_results)
      : wake_parent_mutex_(wake_parent_mutex),
        wake_parent_(wake_parent),
        cancelled_(false),
        running_(false),
        start_index_(start_index),
        end_index_(end_index),
        max_failures_(max_failures),
        num_passes_(0),
        num_failures_(0),
        index_to_input_(index_to_input),
        generate_expected_(generate_expected),
        generate_actual_(generate_actual),
        compare_results_(compare_results),
        ir_text_(ir_text),
        entry_fn_(entry_fn) {}

  // Starts the thread. Silently returns if it's already running.
  // If the tax of calling index_to_input_ every iter is too high, we can
  // specialize this for simple cases, like uint64 -> uint64.
  void Run() {
    if (thread_) {
      return;
    }
    thread_ = absl::make_unique<std::thread>([this]() { RunInternal(); });
  }

  void Join() {
    if (thread_) {
      thread_->join();
    }
    thread_.reset();
  }

  void RunInternal() {
    absl::Status return_status;
    if (cancelled_.load()) {
      return;
    }

    // The spawning Testbench will have already validated the IR text, entry
    // function, and JIT creation, so we can safely call value() here.
    package_ = Parser::ParsePackage(ir_text_).value();
    jit_ = LlvmIrJit::Create(package_->GetFunction(entry_fn_).value()).value();

    auto result_buffer = std::make_unique<uint8[]>(jit_->GetReturnTypeSize());
    absl::Span<uint8> result_span(result_buffer.get(),
                                  jit_->GetReturnTypeSize());

    running_.store(true);
    for (uint64 i = start_index_; i < end_index_; i++) {
      // Don't check for cancelled on every iteration; it's a touch slow.
      if (i % 128 == 0 && cancelled_.load()) {
        return_status = absl::CancelledError("This thread was cancelled.");
        break;
      }

      InputT input = index_to_input_(i);
      ResultT expected = generate_expected_(input);
      ResultT actual = generate_actual_(jit_.get(), result_span, input);
      if (!compare_results_(expected, actual)) {
        num_failures_.store(num_failures_.load() + 1);
        std::string error = absl::StrFormat(
            "Value mismatch at index %d:\n"
            "  Expected: 0x%x\n"
            "  Actual  : 0x%x",
            i, absl::bit_cast<uint32>(expected),
            absl::bit_cast<uint32>(actual));
        XLS_LOG(ERROR) << error;
        if (max_failures_ <= num_failures_.load()) {
          return_status = absl::InternalError(error);
          break;
        }
      } else {
        num_passes_.store(num_passes_.load() + 1);
      }
    }

    running_.store(false);
    {
      absl::MutexLock lock(&mutex_);
      status_ = return_status;
    }
    WakeParent();
  }

  void Cancel() { cancelled_.store(true); }

  bool running() { return running_.load(); }

  uint64 num_failures() { return num_failures_.load(); }

  uint64 num_passes() { return num_passes_.load(); }

  absl::Status status() {
    absl::MutexLock lock(&mutex_);
    return status_;
  }

 private:
  // Kicks the parent threads's condvar to indicate that this thread has
  // finished its work (successfully or otherwise).
  void WakeParent() {
    absl::MutexLock lock(wake_parent_mutex_);
    wake_parent_->Signal();
  }

  // Parent-owned.
  absl::Mutex* wake_parent_mutex_;
  absl::CondVar* wake_parent_;

  // Protects TestbenchThread-owned data.
  absl::Mutex mutex_;

  // The current (and eventually final) status of this worker.
  absl::Status status_ ABSL_GUARDED_BY(mutex_);
  std::atomic<bool> cancelled_;
  std::atomic<bool> running_;

  uint64 start_index_;
  uint64 end_index_;

  // Bookkeeping data.
  uint64 max_failures_;
  std::atomic<uint64> num_passes_;
  std::atomic<uint64> num_failures_;

  std::function<InputT(uint64)> index_to_input_;
  std::function<ResultT(InputT)> generate_expected_;
  std::function<ResultT(LlvmIrJit*, absl::Span<uint8>, InputT)>
      generate_actual_;
  std::function<bool(ResultT, ResultT)> compare_results_;

  std::string ir_text_;
  std::string entry_fn_;
  std::unique_ptr<Package> package_;
  std::unique_ptr<LlvmIrJit> jit_;

  std::unique_ptr<std::thread> thread_;
};

}  // namespace xls

#endif  // XLS_TOOLS_TESTBENCH_THREAD_H_
