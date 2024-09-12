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

#ifndef XLS_TOOLS_TESTBENCH_H_
#define XLS_TOOLS_TESTBENCH_H_

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xls/tests/testbench_thread.h"

namespace xls {

// Testbench is a helper class to test an XLS module (or...anything, really)
// across a range of inputs. This class creates a set of worker threads and
// divides the input across them, and sets them to go until complete.
// Execution status (percent complete, number of result mismatches) will be
// periodically printed to the terminal, as this class' primary use is for
// exploring large test spaces.
//
// Presently, work is parititioned uniformly across threads at startup. This can
// lead to work imbalance if certain areas of the input space execute faster
// than others. More advanced strategies can be explored in the future if this
// becomes a problem.

namespace internal {
// Forward decl of common Testbench base class.
template <typename InputT, typename ResultT, typename ShardDataT>
class TestbenchBase;
}  // namespace internal

// Testbench supports two modes of operation: one that passes per-shard data to
// the execution functions, and one that does not. The implementations switch on
// the presence of the last "ShardDataT" template type parameter. The only real
// difference is the signatures exposed - this implementation has an extra
// ShardDataT parameter on its creation command.
template <typename InputT, typename ResultT, typename ShardDataT = void,
          typename EnableT = void>
class Testbench : public internal::TestbenchBase<InputT, ResultT, ShardDataT> {
 public:
  // Args:
  //   start, end      : The bounds of the space to evaluate, as [start, end).
  //   max_failures    : The maximum number of result mismatches to allow (per
  //                     worker thread) before cancelling execution.
  //   index_to_input  : The function to call to convert an index (uint64_t) to
  //                     an input (InputT).
  //   compute_expected: The function to calculate the expected result.
  //   compute_actual  : The function to call to calculate the XLS result.
  //                     "result_buffer" is a convenience buffer provided as
  //                     isn't directly used internally all - it's just a
  //                     convenience to avoid the need to heap allocate on every
  //                     iteration.
  //   compare_results : Should return true if both ResultTs (expected & actual)
  //                     are considered equivalent.
  //   log_errors      : The function to log errors when compare_results returns
  //                     false.
  //
  // All lambdas must be thread-safe.
  //
  // "compute_expected" and "compute_actual" return pure InputTs and ResultTs
  // instead of wrapping them in StatusOrs so we don't pay that tax on every
  // iteration. If our algorithms die, we should fix that before evaluating for
  // correctness (since any changes might affect results).
  Testbench(uint64_t start, uint64_t end, int64_t num_threads,
            uint64_t max_failures,
            std::function<InputT(uint64_t)> index_to_input,
            std::function<std::unique_ptr<ShardDataT>()> create_shard,
            std::function<ResultT(ShardDataT*, InputT)> compute_expected,
            std::function<ResultT(ShardDataT*, InputT)> compute_actual,
            std::function<bool(ResultT, ResultT)> compare_results,
            std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors)
      : internal::TestbenchBase<InputT, ResultT, ShardDataT>(
            start, end, num_threads, max_failures, index_to_input,
            compare_results, log_errors),
        create_shard_(create_shard),
        compute_expected_(compute_expected),
        compute_actual_(compute_actual) {
    this->thread_create_fn_ = [this](uint64_t start, uint64_t end) {
      return std::make_unique<TestbenchThread<InputT, ResultT, ShardDataT>>(
          &this->mutex_, &this->wake_me_, start, end, this->max_failures_,
          this->index_to_input_, create_shard_, compute_expected_,
          compute_actual_, this->compare_results_, this->log_errors_);
    };
  }

 private:
  std::function<std::unique_ptr<ShardDataT>()> create_shard_;
  std::function<ResultT(ShardDataT*, InputT)> compute_expected_;
  std::function<ResultT(ShardDataT*, InputT)> compute_actual_;
};

// Shard-data-less implementation.
template <typename InputT, typename ResultT, typename ShardDataT>
class Testbench<InputT, ResultT, ShardDataT,
                typename std::enable_if<std::is_void<ShardDataT>::value>::type>
    : public internal::TestbenchBase<InputT, ResultT, ShardDataT> {
 public:
  Testbench(uint64_t start, uint64_t end, int64_t num_threads,
            uint64_t max_failures,
            std::function<InputT(uint64_t)> index_to_input,
            std::function<ResultT(InputT)> compute_expected,
            std::function<ResultT(InputT)> compute_actual,
            std::function<bool(ResultT, ResultT)> compare_results,
            std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors)
      : internal::TestbenchBase<InputT, ResultT, ShardDataT>(
            start, end, num_threads, max_failures, index_to_input,
            compare_results, log_errors),
        compute_expected_(compute_expected),
        compute_actual_(compute_actual) {
    this->thread_create_fn_ = [this](uint64_t start, uint64_t end) {
      return std::make_unique<TestbenchThread<InputT, ResultT, ShardDataT>>(
          &this->mutex_, &this->wake_me_, start, end, this->max_failures_,
          this->index_to_input_, compute_expected_, compute_actual_,
          this->compare_results_, this->log_errors_);
    };
  }

 private:
  std::function<ResultT(InputT)> compute_expected_;
  std::function<ResultT(InputT)> compute_actual_;
};

// INTERNAL IMPL ---------------------------------
namespace internal {

// This common base class implements the _real_ logic: spawning runner threads
// and monitoring the results.
template <typename InputT, typename ResultT, typename ShardDataT>
class TestbenchBase {
 public:
  TestbenchBase(
      uint64_t start, uint64_t end, uint64_t num_threads, uint64_t max_failures,
      std::function<InputT(uint64_t)> index_to_input,
      std::function<bool(ResultT, ResultT)> compare_results,
      std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors)
      : started_(false),
        num_threads_(num_threads),
        start_(start),
        end_(end),
        max_failures_(max_failures),
        num_samples_processed_(0),
        index_to_input_(index_to_input),
        compare_results_(compare_results),
        log_errors_(log_errors) {}

  // Sets the number of threads to use. Must be called before Run().
  absl::Status SetNumThreads(int num_threads) {
    absl::MutexLock lock(&mutex_);
    if (this->started_) {
      return absl::FailedPreconditionError(
          "Can't change the number of threads after starting execution.");
    }
    num_threads_ = num_threads;
    return absl::OkStatus();
  }

  // Executes the test.
  absl::Status Run() {
    // Lock before spawning threads to prevent missing any early wakeup signals
    // here.
    mutex_.Lock();
    started_ = true;

    // Set up all the workers.
    uint64_t chunk_size = (end_ - start_) / num_threads_;
    uint64_t chunk_remainder =
        chunk_size == 0 ? (end_ - start_) : (end_ - start_) % chunk_size;
    uint64_t first = 0;
    uint64_t last;
    for (int i = 0; i < num_threads_; i++) {
      last = first + chunk_size;
      // Distribute any remainder evenly amongst the threads.
      if (chunk_remainder > 0) {
        last++;
        chunk_remainder--;
      }

      threads_.push_back(thread_create_fn_(first, last));
      threads_.back()->Run();

      first = last + 1;
    }

    // Wait for all to be ready.
    bool all_ready = false;
    while (!all_ready) {
      int64_t num_ready = 0;
      wake_me_.Wait(&mutex_);

      for (int64_t i = 0; i < threads_.size(); i++) {
        if (threads_[i]->ready()) {
          num_ready++;
        }
      }
      all_ready = num_ready == threads_.size();
    }

    // Don't include startup time.
    start_time_ = absl::Now();

    for (int i = 0; i < threads_.size(); i++) {
      threads_[i]->SignalStart();
    }

    // Now monitor them.
    bool done = false;
    while (!done) {
      int64_t num_done = 0;
      wake_me_.WaitWithTimeout(&mutex_, kPrintInterval);

      PrintStatus();

      // See if everyone's done or if someone blew up.
      for (int64_t i = 0; i < threads_.size(); i++) {
        if (!threads_[i]->running()) {
          num_done++;
          absl::Status status = threads_[i]->status();
          if (!status.ok()) {
            Cancel();
            num_done = threads_.size();
            break;
          }
        }
      }

      done = num_done == threads_.size();
    }

    // When exiting the loop, we'll be holding the lock (due to
    // WaitWithTimeout).
    mutex_.Unlock();

    // Join threads at the end because we are polite.
    for (int i = 0; i < threads_.size(); i++) {
      threads_[i]->Join();
    }

    for (int i = 0; i < threads_.size(); i++) {
      if (threads_[i]->num_failures() != 0) {
        return absl::InternalError(
            "There was at least one mismatch during execution.");
      }
    }

    return absl::OkStatus();
  }

 protected:
  // How many seconds to wait before printing status (at most).
  static constexpr absl::Duration kPrintInterval = absl::Seconds(5);

  // Prints the current execution status across all threads.
  void PrintStatus() {
    // Get the remainder-adjusted chunk size for this thread.
    auto thread_chunk_size = [this](int thread_index) {
      uint64_t total_size = end_ - start_;
      uint64_t chunk_size = total_size / threads_.size();
      uint64_t remainder =
          chunk_size == 0 ? total_size : total_size % chunk_size;
      if (thread_index < remainder) {
        chunk_size++;
      }
      return chunk_size;
    };

    // Ignore remainder issues here. It shouldn't matter much at all.
    absl::Time now = absl::Now();
    auto delta = now - start_time_;
    uint64_t total_done = 0;
    for (int64_t i = 0; i < threads_.size(); ++i) {
      uint64_t num_passes = threads_[i]->num_passes();
      uint64_t num_failures = threads_[i]->num_failures();
      uint64_t thread_done = num_passes + num_failures;
      uint64_t chunk_size = thread_chunk_size(i);
      total_done += thread_done;
      std::cout << absl::StreamFormat(
                       "thread %02d: %f%% @ %.1f us/sample :: failures %d", i,
                       static_cast<double>(thread_done) / chunk_size * 100.0,
                       absl::ToDoubleMicroseconds(delta) / thread_done,
                       num_failures)
                << "\n";
    }
    double done_per_second = delta == absl::ZeroDuration()
                                 ? 0.0
                                 : total_done / absl::ToDoubleSeconds(delta);
    int64_t remaining = end_ - start_ - total_done;
    auto estimate = absl::Seconds(
        done_per_second == 0.0 ? 0.0 : remaining / done_per_second);
    double throughput_this_print =
        static_cast<double>(total_done - num_samples_processed_) /
        ToInt64Seconds(kPrintInterval);
    std::cout << absl::StreamFormat(
                     "--- ^ after %s elapsed; %f Misamples/s; estimate %s "
                     "remaining ...",
                     absl::FormatDuration(delta),
                     throughput_this_print / std::pow(2, 20),
                     absl::FormatDuration(estimate))
              << std::endl;
    num_samples_processed_ = total_done;
  }

  // Requests that all running threads terminate (but doesn't Join() them).
  void Cancel() {
    for (int i = 0; i < threads_.size(); i++) {
      threads_[i]->Cancel();
    }
  }

  bool started_;
  int num_threads_;
  absl::Time start_time_;
  uint64_t start_;
  uint64_t end_;
  uint64_t max_failures_;
  uint64_t num_samples_processed_;
  std::function<InputT(uint64_t)> index_to_input_;
  std::function<bool(ResultT, ResultT)> compare_results_;
  std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors_;

  using ThreadT = TestbenchThread<InputT, ResultT, ShardDataT>;
  std::function<std::unique_ptr<ThreadT>(uint64_t, uint64_t)> thread_create_fn_;
  std::vector<std::unique_ptr<ThreadT>> threads_;

  // The main thread sleeps while tests are running. As worker threads finish,
  // they'll wake us up via this condvar.
  absl::Mutex mutex_;
  absl::CondVar wake_me_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal
}  // namespace xls

#endif  // XLS_TOOLS_TESTBENCH_H_
