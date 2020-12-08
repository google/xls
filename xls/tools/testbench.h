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

#include <functional>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>

#include "absl/base/internal/sysinfo.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/integral_types.h"
#include "xls/tools/testbench_thread.h"

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
template <typename JitWrapperT, typename InputT, typename ResultT,
          typename ShardDataT>
class TestbenchBase;
}  // namespace internal

// Testbench supports two modes of operation: one that passes per-shard data to
// the execution functions, and one that does not. The implementations switch on
// the presence of the last "ShardDataT" template type parameter. The only real
// difference is the signatures exposed - this implementation has an extra
// ShardDataT parameter on its creation command.
template <typename JitWrapperT, typename InputT, typename ResultT,
          typename ShardDataT = void, typename EnableT = void>
class Testbench
    : public internal::TestbenchBase<JitWrapperT, InputT, ResultT, ShardDataT> {
 public:
  // Args:
  //   start, end: The bounds of the space to evaluate, as [start, end).
  //   max_failures: The maximum number of result mismatches to allow (per
  //                  worker thread) before cancelling execution.
  //   compute_actual: The function to call to calculate the XLS result.
  //     "result_buffer" is a convenience buffer provided as temporary storage
  //     to hold result data if using view types in the calculation. This buffer
  //     isn't directly used internally all - it's just a convenience to avoid
  //     the need to heap allocate on every iteration.
  //   compare_results: Should return true if both ResultTs (expected & actual)
  //                     are considered equivalent.
  // These lambdas return pure InputTs and ResultTs instead of wrapping them in
  // StatusOrs so we don't pay that tax on every iteration. If our algorithms
  // die, we should fix that before evaluating for correctness (since any
  // changes might affect results).
  // All lambdas must be thread-safe.
  Testbench(
      uint64 start, uint64 end, uint64 max_failures,
      std::function<InputT(uint64)> index_to_input,
      std::function<std::unique_ptr<ShardDataT>()> create_shard,
      std::function<ResultT(ShardDataT*, InputT)> compute_expected,
      std::function<ResultT(JitWrapperT*, ShardDataT*, InputT)> compute_actual,
      std::function<bool(ResultT, ResultT)> compare_results)
      : internal::TestbenchBase<JitWrapperT, InputT, ResultT, ShardDataT>(
            start, end, max_failures, index_to_input, compare_results),
        create_shard_(create_shard),
        compute_expected_(compute_expected),
        compute_actual_(compute_actual) {
    this->thread_create_fn_ = [this](uint64 start, uint64 end) {
      return std::make_unique<
          TestbenchThread<JitWrapperT, InputT, ResultT, ShardDataT>>(
          &this->mutex_, &this->wake_me_, start, end, this->max_failures_,
          this->index_to_input_, create_shard_, compute_expected_,
          compute_actual_, this->compare_results_);
    };
  }

 private:
  std::function<std::unique_ptr<ShardDataT>()> create_shard_;
  std::function<ResultT(ShardDataT*, InputT)> compute_expected_;
  std::function<ResultT(JitWrapperT*, ShardDataT*, InputT)> compute_actual_;
};

// Shard-data-less implementation.
template <typename JitWrapperT, typename InputT, typename ResultT,
          typename ShardDataT>
class Testbench<JitWrapperT, InputT, ResultT, ShardDataT,
                typename std::enable_if<std::is_void<ShardDataT>::value>::type>
    : public internal::TestbenchBase<JitWrapperT, InputT, ResultT, ShardDataT> {
 public:
  Testbench(uint64 start, uint64 end, uint64 max_failures,
            std::function<InputT(uint64)> index_to_input,
            std::function<ResultT(InputT)> compute_expected,
            std::function<ResultT(JitWrapperT*, InputT)> compute_actual,
            std::function<bool(ResultT, ResultT)> compare_results)
      : internal::TestbenchBase<JitWrapperT, InputT, ResultT, ShardDataT>(
            start, end, max_failures, index_to_input, compare_results),
        compute_expected_(compute_expected),
        compute_actual_(compute_actual) {
    this->thread_create_fn_ = [this](uint64 start, uint64 end) {
      return std::make_unique<
          TestbenchThread<JitWrapperT, InputT, ResultT, ShardDataT>>(
          &this->mutex_, &this->wake_me_, start, end, this->max_failures_,
          this->index_to_input_, compute_expected_, compute_actual_,
          this->compare_results_);
    };
  }

 private:
  std::function<ResultT(InputT)> compute_expected_;
  std::function<ResultT(JitWrapperT*, InputT)> compute_actual_;
};

// INTERNAL IMPL ---------------------------------
namespace internal {

// This common base class implements the _real_ logic: spawning runner threads
// and monitoring the results.
template <typename JitWrapperT, typename InputT, typename ResultT,
          typename ShardDataT>
class TestbenchBase {
 public:
  TestbenchBase(uint64 start, uint64 end, uint64 max_failures,
                std::function<InputT(uint64)> index_to_input,
                std::function<bool(ResultT, ResultT)> compare_results)
      : started_(false),
        num_threads_(std::thread::hardware_concurrency()),
        start_(start),
        end_(end),
        max_failures_(max_failures),
        num_samples_processed_(0),
        index_to_input_(index_to_input),
        compare_results_(compare_results) {}

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
    start_time_ = absl::Now();

    // Set up all the workers.
    uint64 chunk_size = (end_ - start_) / num_threads_;
    uint64 chunk_remainder = (end_ - start_) % chunk_size;
    uint64 first = 0;
    uint64 last;
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

    // Now monitor them.
    bool done = false;
    while (!done) {
      int num_done = 0;
      wake_me_.WaitWithTimeout(&mutex_, kPrintInterval);

      PrintStatus();

      // See if everyone's done or if someone blew up.
      for (int i = 0; i < threads_.size(); i++) {
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
      uint64 total_size = end_ - start_;
      uint64 chunk_size = total_size / threads_.size();
      uint64 remainder = total_size % chunk_size;
      if (thread_index < remainder) {
        chunk_size++;
      }
      return chunk_size;
    };

    // Ignore remainder issues here. It shouldn't matter much at all.
    absl::Time now = absl::Now();
    auto delta = now - start_time_;
    uint64 total_done = 0;
    for (int64 i = 0; i < threads_.size(); ++i) {
      uint64 num_passes = threads_[i]->num_passes();
      uint64 num_failures = threads_[i]->num_failures();
      uint64 thread_done = num_passes + num_failures;
      uint64 chunk_size = thread_chunk_size(i);
      total_done += thread_done;
      std::cout << absl::StreamFormat(
                       "thread %02d: %f%% @ %.1f us/sample :: failures %d", i,
                       static_cast<double>(thread_done) / chunk_size * 100.0,
                       absl::ToDoubleMicroseconds(delta) / thread_done,
                       num_failures)
                << "\n";
    }
    double done_per_second = total_done / absl::ToDoubleSeconds(delta);
    int64 remaining = end_ - start_ - total_done;
    auto estimate = absl::Seconds(remaining / done_per_second);
    double throughput_this_print =
        static_cast<double>(total_done - num_samples_processed_) /
        ToInt64Seconds(kPrintInterval);
    std::cout << absl::StreamFormat(
                     "--- ^ after %s elapsed; %.2f Misamples/s; estimate %s "
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
  uint64 start_;
  uint64 end_;
  uint64 max_failures_;
  uint64 num_samples_processed_;
  std::function<InputT(uint64)> index_to_input_;
  std::function<bool(ResultT, ResultT)> compare_results_;

  using ThreadT = TestbenchThread<JitWrapperT, InputT, ResultT, ShardDataT>;
  std::function<std::unique_ptr<ThreadT>(uint64, uint64)> thread_create_fn_;
  std::vector<std::unique_ptr<ThreadT>> threads_;

  // The main thread sleeps while tests are running. As worker threads finish,
  // they'll wake us up via this condvar.
  absl::Mutex mutex_;
  absl::CondVar wake_me_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal
}  // namespace xls

#endif  // XLS_TOOLS_TESTBENCH_H_
