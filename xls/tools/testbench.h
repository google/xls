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

#ifndef XLS_TOOLS_TESTBENCH_H_
#define XLS_TOOLS_TESTBENCH_H_

#include <functional>

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
template <typename InputT, typename ResultT>
class Testbench {
 public:
  //  - start, end: The bounds of the space to evaluate, as [start, end).
  //  - max_failures: The maximum number of result mismatches to allow (per
  //                  worker thread) before cancelling execution.
  //  - compare_results: Should return true if both ResultTs (expected & actual)
  //                     are considered equivalent.
  // These lambdas return pure InputTs and ResultTs instead of wrapping them in
  // StatusOrs so we don't pay that tax on every iteration. If our algorithms
  // die, we should fix that before evaluating for correctness (since any
  // changes might affect results).
  // All lambdas must be thread-safe.
  Testbench(std::string ir_path, std::string entry_function, uint64 start,
            uint64 end, uint64 max_failures,
            std::function<InputT(uint64)> index_to_input,
            std::function<ResultT(InputT)> compute_expected,
            std::function<ResultT(LlvmIrJit*, absl::Span<uint8>, InputT)>
                compute_actual,
            std::function<bool(ResultT, ResultT)> compare_results);

  // Sets the number of threads to use. Must be called before Run().
  absl::Status SetNumThreads(int num_threads) {
    absl::MutexLock lock(&mutex_);
    if (started_) {
      return absl::FailedPreconditionError(
          "Can't change the number of threads after starting execution.");
    }
    num_threads_ = num_threads;
    return absl::OkStatus();
  }

  // Executes the test.
  absl::Status Run();

 private:
  // How many seconds to wait before printing status (at most).
  static constexpr absl::Duration kPrintInterval = absl::Seconds(5);

  // Requests that all running threads terminate (but doesn't Join() them).
  void Cancel();

  // Prints the current execution status across all threads.
  void PrintStatus();

  // The main thread sleeps while tests are running. As worker threads finish,
  // they'll wake us up via this condvar.
  absl::Mutex mutex_;
  absl::CondVar wake_me_;

  std::vector<std::unique_ptr<TestbenchThread<InputT, ResultT>>> threads_;

  bool started_;
  int num_threads_;
  absl::Time start_time_;
  uint64 start_;
  uint64 end_;
  uint64 max_failures_;
  uint64 num_samples_processed_;
  std::function<InputT(uint64)> index_to_input_;
  std::function<ResultT(InputT)> compute_expected_;
  std::function<ResultT(LlvmIrJit*, absl::Span<uint8>, InputT)> compute_actual_;
  std::function<bool(ResultT, ResultT)> compare_results_;

  std::string ir_path_;
  std::string entry_function_;
};

// INTERNAL IMPL ---------------------------------

template <typename InputT, typename ResultT>
Testbench<InputT, ResultT>::Testbench(
    std::string ir_path, std::string entry_function, uint64 start, uint64 end,
    uint64 max_failures, std::function<InputT(uint64)> index_to_input,
    std::function<ResultT(InputT)> compute_expected,
    std::function<ResultT(LlvmIrJit*, absl::Span<uint8>, InputT)>
        compute_actual,
    std::function<bool(ResultT, ResultT)> compare_results)
    : started_(false),
      num_threads_(absl::base_internal::NumCPUs()),
      start_(start),
      end_(end),
      max_failures_(max_failures),
      num_samples_processed_(0),
      index_to_input_(index_to_input),
      compute_expected_(compute_expected),
      compute_actual_(compute_actual),
      compare_results_(compare_results),
      ir_path_(std::move(ir_path)),
      entry_function_(std::move(entry_function)) {}

template <typename InputT, typename ResultT>
absl::Status Testbench<InputT, ResultT>::Run() {
  // First, verify that the IR and entry function are sane to avoid needing to
  // check in the threads (one error message is nicer than 20).
  // Reminder: we don't use this Package for anything; each thread needs its own
  // due to thread-safety concerns.
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path_));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text));
  XLS_RETURN_IF_ERROR(package->GetFunction(entry_function_).status());

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

    threads_.push_back(std::make_unique<TestbenchThread<InputT, ResultT>>(
        ir_text, entry_function_, &mutex_, &wake_me_, first, last,
        max_failures_, index_to_input_, compute_expected_, compute_actual_,
        compare_results_));
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

  // When exiting the loop, we'll be holding the lock (due to WaitWithTimeout).
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

template <typename InputT, typename ResultT>
void Testbench<InputT, ResultT>::PrintStatus() {
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

template <typename InputT, typename ResultT>
void Testbench<InputT, ResultT>::Cancel() {
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i]->Cancel();
  }
}

}  // namespace xls

#endif  // XLS_TOOLS_TESTBENCH_H_
