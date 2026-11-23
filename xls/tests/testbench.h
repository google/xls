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

#ifndef XLS_TESTS_TESTBENCH_H_
#define XLS_TESTS_TESTBENCH_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

namespace xls {

// Testbench is a helper class to test an XLS module (or...anything, really)
// across a range of inputs. This class creates a set of worker threads,
// divides the input range across them, and periodically prints execution
// progress (percent complete, number of result mismatches) to stdout.
//
// Presently, work is partitioned uniformly across threads at startup. This can
// lead to work imbalance if certain areas of the input space execute faster
// than others. More advanced strategies (e.g., atomic chunk claiming) can be
// explored in the future if this becomes a problem.

namespace internal {

// Cache-line aligned per-worker progress counters to avoid false sharing.
struct alignas(ABSL_CACHELINE_SIZE) WorkerProgress {
  std::atomic<uint64_t> passes{0};
  std::atomic<uint64_t> failures{0};
  uint64_t chunk_size = 0;
};

// Non-templated base implementing thread lifecycle, range partitioning,
// startup barrier, progress logging, and SetNumThreads synchronization.
class TestbenchRunner {
 public:
  TestbenchRunner(uint64_t start, uint64_t end, int64_t num_threads,
                  uint64_t max_failures);

  // Sets the number of threads to use. Must be called before Run().
  absl::Status SetNumThreads(int64_t num_threads);

 protected:
  // Passed to each spawned worker: handles shard-local initialization, waits
  // for all siblings via `wait_for_start()` (returns false if cancelled before
  // starting), then evaluates indices in the half-open range `[first, last)`.
  using WorkerFn = absl::FunctionRef<void(
      uint64_t first, uint64_t last, WorkerProgress& progress,
      std::atomic<bool>& cancelled, absl::FunctionRef<bool()> wait_for_start)>;

  static constexpr uint64_t kCancelCheckIterationInterval = 128;

  ~TestbenchRunner() = default;

  // Partitions `[start_, end_)` across worker threads, waits for all workers
  // to finish initializing before starting the throughput clock, prints
  // periodic progress updates, and returns an error if any worker reached
  // `max_failures_`.
  absl::Status RunWorkers(WorkerFn worker_fn);

  template <typename InputT, typename ResultT, typename ExpectedFn,
            typename ActualFn>
  void RunRange(uint64_t first, uint64_t last, WorkerProgress& progress,
                std::atomic<bool>& cancelled,
                const std::function<InputT(uint64_t)>& index_to_input,
                ExpectedFn&& compute_expected, ActualFn&& compute_actual,
                const std::function<bool(ResultT, ResultT)>& compare_results,
                const std::function<void(int64_t, InputT, ResultT, ResultT)>&
                    log_errors) const {
    uint64_t local_passes = 0;
    uint64_t local_failures = 0;
    for (uint64_t i = first; i < last; ++i) {
      // Avoid polling the shared atomic on every iteration in tight loops.
      if ((i - first) % kCancelCheckIterationInterval == 0 &&
          cancelled.load(std::memory_order_relaxed)) {
        break;
      }

      InputT input = index_to_input(i);
      ResultT expected = compute_expected(input);
      ResultT actual = compute_actual(input);
      if (!compare_results(expected, actual)) {
        ++local_failures;
        progress.failures.store(local_failures, std::memory_order_relaxed);
        log_errors(static_cast<int64_t>(i), input, expected, actual);
        if (max_failures_ != 0 && local_failures >= max_failures_) {
          cancelled.store(true, std::memory_order_relaxed);
          break;
        }
      } else {
        ++local_passes;
        progress.passes.store(local_passes, std::memory_order_relaxed);
      }
    }
  }

 private:
  absl::Mutex mutex_;
  bool started_ ABSL_GUARDED_BY(mutex_) = false;
  int64_t num_threads_ ABSL_GUARDED_BY(mutex_);
  uint64_t start_;
  uint64_t end_;
  uint64_t max_failures_;
};

}  // namespace internal

// Testbench supports two modes of operation: one that passes per-shard data to
// the execution functions, and one that does not. The implementations switch on
// the presence of the last "ShardDataT" template type parameter. The only real
// difference is the signatures exposed - this implementation has an extra
// ShardDataT parameter on its creation command.
template <typename InputT, typename ResultT, typename ShardDataT = void,
          typename EnableT = void>
class Testbench : public internal::TestbenchRunner {
 public:
  // Args:
  //   start, end      : The bounds of the space to evaluate, as [start, end).
  //   num_threads     : Maximum worker threads to run in parallel.
  //   max_failures    : The maximum number of result mismatches to allow (per
  //                     worker thread) before cancelling execution (0 = no
  //                     limit).
  //   index_to_input  : Converts an index (uint64_t) to an input (InputT).
  //   create_shard    : Creates per-worker state once on each worker thread.
  //   compute_expected: Calculates the expected result.
  //   compute_actual  : Calculates the result under test.
  //   compare_results : Returns true if expected and actual match.
  //   log_errors      : Logs mismatches when compare_results returns false.
  //
  // All lambdas must be thread-safe.
  //
  // "compute_expected" and "compute_actual" return pure ResultTs instead of
  // wrapping them in StatusOrs so we don't pay that tax on every iteration.
  Testbench(uint64_t start, uint64_t end, int64_t num_threads,
            uint64_t max_failures,
            std::function<InputT(uint64_t)> index_to_input,
            std::function<std::unique_ptr<ShardDataT>()> create_shard,
            std::function<ResultT(ShardDataT*, InputT)> compute_expected,
            std::function<ResultT(ShardDataT*, InputT)> compute_actual,
            std::function<bool(ResultT, ResultT)> compare_results,
            std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors)
      : internal::TestbenchRunner(start, end, num_threads, max_failures),
        index_to_input_(std::move(index_to_input)),
        create_shard_(std::move(create_shard)),
        compute_expected_(std::move(compute_expected)),
        compute_actual_(std::move(compute_actual)),
        compare_results_(std::move(compare_results)),
        log_errors_(std::move(log_errors)) {}

  // Executes the test across [start, end). May only be called once.
  absl::Status Run() {
    return RunWorkers([this](uint64_t first, uint64_t last,
                             internal::WorkerProgress& progress,
                             std::atomic<bool>& cancelled,
                             absl::FunctionRef<bool()> wait_for_start) {
      std::unique_ptr<ShardDataT> shard = create_shard_();
      if (!wait_for_start()) {
        return;
      }
      RunRange<InputT, ResultT>(
          first, last, progress, cancelled, index_to_input_,
          [this, shard_ptr = shard.get()](const InputT& input) {
            return compute_expected_(shard_ptr, input);
          },
          [this, shard_ptr = shard.get()](const InputT& input) {
            return compute_actual_(shard_ptr, input);
          },
          compare_results_, log_errors_);
    });
  }

 private:
  std::function<InputT(uint64_t)> index_to_input_;
  std::function<std::unique_ptr<ShardDataT>()> create_shard_;
  std::function<ResultT(ShardDataT*, InputT)> compute_expected_;
  std::function<ResultT(ShardDataT*, InputT)> compute_actual_;
  std::function<bool(ResultT, ResultT)> compare_results_;
  std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors_;
};

// Shard-data-less implementation.
template <typename InputT, typename ResultT, typename ShardDataT>
class Testbench<InputT, ResultT, ShardDataT,
                std::enable_if_t<std::is_void_v<ShardDataT>>>
    : public internal::TestbenchRunner {
 public:
  Testbench(uint64_t start, uint64_t end, int64_t num_threads,
            uint64_t max_failures,
            std::function<InputT(uint64_t)> index_to_input,
            std::function<ResultT(InputT)> compute_expected,
            std::function<ResultT(InputT)> compute_actual,
            std::function<bool(ResultT, ResultT)> compare_results,
            std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors)
      : internal::TestbenchRunner(start, end, num_threads, max_failures),
        index_to_input_(std::move(index_to_input)),
        compute_expected_(std::move(compute_expected)),
        compute_actual_(std::move(compute_actual)),
        compare_results_(std::move(compare_results)),
        log_errors_(std::move(log_errors)) {}

  // Executes the test across [start, end). May only be called once.
  absl::Status Run() {
    return RunWorkers([this](uint64_t first, uint64_t last,
                             internal::WorkerProgress& progress,
                             std::atomic<bool>& cancelled,
                             absl::FunctionRef<bool()> wait_for_start) {
      if (!wait_for_start()) {
        return;
      }
      RunRange<InputT, ResultT>(first, last, progress, cancelled,
                                index_to_input_, compute_expected_,
                                compute_actual_, compare_results_, log_errors_);
    });
  }

 private:
  std::function<InputT(uint64_t)> index_to_input_;
  std::function<ResultT(InputT)> compute_expected_;
  std::function<ResultT(InputT)> compute_actual_;
  std::function<bool(ResultT, ResultT)> compare_results_;
  std::function<void(int64_t, InputT, ResultT, ResultT)> log_errors_;
};

}  // namespace xls

#endif  // XLS_TESTS_TESTBENCH_H_
