// Copyright 2026 The XLS Authors
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

#include "xls/tests/testbench.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/stopwatch.h"
#include "xls/common/thread.h"

namespace xls::internal {
namespace {

constexpr absl::Duration kPrintInterval = absl::Seconds(5);
constexpr double kSamplesPerMiSample = 1024.0 * 1024.0;

struct PrintState {
  absl::Duration last_elapsed = absl::ZeroDuration();
  uint64_t last_total_done = 0;
};

void PrintStatus(absl::Span<const WorkerProgress> progress, uint64_t total_size,
                 const Stopwatch& stopwatch, PrintState& state) {
  const absl::Duration elapsed = stopwatch.GetElapsedTime();
  const absl::Duration interval = elapsed - state.last_elapsed;
  uint64_t total_done = 0;
  int64_t thread_index = 0;
  for (const WorkerProgress& worker : progress) {
    const uint64_t num_passes = worker.passes.load(std::memory_order_relaxed);
    const uint64_t num_failures =
        worker.failures.load(std::memory_order_relaxed);
    const uint64_t thread_done = num_passes + num_failures;
    const uint64_t chunk_size = worker.chunk_size;
    total_done += thread_done;
    const double pct = chunk_size == 0
                           ? 100.0
                           : static_cast<double>(thread_done) /
                                 static_cast<double>(chunk_size) * 100.0;
    const double us_per_sample = thread_done == 0
                                     ? 0.0
                                     : absl::ToDoubleMicroseconds(elapsed) /
                                           static_cast<double>(thread_done);
    std::cout << absl::StreamFormat(
        "thread %02d: %f%% @ %.1f us/sample :: failures %d\n", thread_index,
        pct, us_per_sample, num_failures);
    ++thread_index;
  }
  const double elapsed_sec = absl::ToDoubleSeconds(elapsed);
  const double done_per_second =
      elapsed_sec <= 0.0 ? 0.0 : static_cast<double>(total_done) / elapsed_sec;
  const uint64_t remaining =
      total_done >= total_size ? 0 : total_size - total_done;
  const absl::Duration estimate =
      absl::Seconds(done_per_second == 0.0
                        ? 0.0
                        : static_cast<double>(remaining) / done_per_second);
  const double interval_sec = absl::ToDoubleSeconds(interval);
  const uint64_t delta_done = total_done > state.last_total_done
                                  ? total_done - state.last_total_done
                                  : 0;
  const double throughput_this_print =
      interval_sec <= 0.0 ? 0.0
                          : static_cast<double>(delta_done) / interval_sec;
  std::cout << absl::StreamFormat(
                   "--- ^ after %s elapsed; %f Misamples/s; estimate %s "
                   "remaining ...\n",
                   absl::FormatDuration(elapsed),
                   throughput_this_print / kSamplesPerMiSample,
                   absl::FormatDuration(estimate))
            << std::flush;
  state.last_elapsed = elapsed;
  state.last_total_done = total_done;
}

}  // namespace

TestbenchRunner::TestbenchRunner(uint64_t start, uint64_t end,
                                 int64_t num_threads, uint64_t max_failures)
    : num_threads_(num_threads),
      start_(start),
      end_(end),
      max_failures_(max_failures) {}

absl::Status TestbenchRunner::SetNumThreads(int64_t num_threads) {
  absl::MutexLock lock(mutex_);
  if (started_) {
    return absl::FailedPreconditionError(
        "Can't change the number of threads after starting execution.");
  }
  if (num_threads <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("num_threads must be positive; got %d.", num_threads));
  }
  num_threads_ = num_threads;
  return absl::OkStatus();
}

absl::Status TestbenchRunner::RunWorkers(WorkerFn worker_fn) {
  if (start_ > end_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid testbench range: [%u, %u).", start_, end_));
  }

  int64_t num_threads = 0;
  {
    absl::MutexLock lock(mutex_);
    if (started_) {
      return absl::FailedPreconditionError(
          "Testbench has already been started.");
    }
    if (num_threads_ <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "num_threads must be positive; got %d.", num_threads_));
    }
    started_ = true;
    num_threads = num_threads_;
  }

  const uint64_t total_size = end_ - start_;
  if (total_size == 0) {
    return absl::OkStatus();
  }

  // BlockingCounter takes an `int` initial count.
  const int64_t active_threads = static_cast<int64_t>(std::min(
      static_cast<uint64_t>(num_threads),
      std::min(total_size,
               static_cast<uint64_t>(std::numeric_limits<int>::max()))));

  // Declare synchronization state before `workers` so ~Thread() joins every
  // worker before any referenced local is destroyed.
  std::vector<WorkerProgress> progress(static_cast<size_t>(active_threads));
  std::atomic<bool> cancelled{false};
  absl::BlockingCounter ready(static_cast<int>(active_threads));
  absl::Notification start_signal;
  std::atomic<int64_t> remaining{active_threads};
  absl::Notification all_done;

  std::vector<std::unique_ptr<Thread>> workers;
  workers.reserve(static_cast<size_t>(active_threads));

  // If unwinding before `start_signal` fires, unblock waiting workers before
  // `workers` joins them.
  absl::Cleanup release_waiting_workers = [&cancelled, &start_signal]() {
    if (!start_signal.HasBeenNotified()) {
      cancelled.store(true, std::memory_order_relaxed);
      start_signal.Notify();
    }
  };

  const uint64_t num_workers = static_cast<uint64_t>(active_threads);
  const uint64_t chunk_size = total_size / num_workers;
  uint64_t chunk_remainder = total_size % num_workers;
  uint64_t first = start_;
  for (int64_t i = 0; i < active_threads; ++i) {
    uint64_t count = chunk_size;
    if (chunk_remainder > 0) {
      ++count;
      --chunk_remainder;
    }
    const uint64_t last = first + count;
    const size_t worker_idx = static_cast<size_t>(i);
    progress[worker_idx].chunk_size = count;
    workers.push_back(std::make_unique<Thread>(
        [&progress, &cancelled, &ready, &start_signal, &remaining, &all_done,
         worker_fn, worker_idx, first, last]() {
          bool ready_signaled = false;
          absl::Cleanup finish_worker = [&ready_signaled, &ready, &remaining,
                                         &all_done]() {
            if (!ready_signaled) {
              ready.DecrementCount();
            }
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
              all_done.Notify();
            }
          };
          auto wait_for_start = [&ready_signaled, &ready, &start_signal,
                                 &cancelled]() -> bool {
            ready_signaled = true;
            ready.DecrementCount();
            start_signal.WaitForNotification();
            return !cancelled.load(std::memory_order_relaxed);
          };
          worker_fn(first, last, progress[worker_idx], cancelled,
                    wait_for_start);
        }));
    first = last;
  }

  // Exclude shard initialization time from throughput stats.
  ready.Wait();
  const Stopwatch stopwatch;
  start_signal.Notify();

  PrintState print_state;
  while (!all_done.WaitForNotificationWithTimeout(kPrintInterval)) {
    PrintStatus(progress, total_size, stopwatch, print_state);
  }

  for (const std::unique_ptr<Thread>& worker : workers) {
    worker->Join();
  }
  PrintStatus(progress, total_size, stopwatch, print_state);

  if (max_failures_ != 0) {
    for (const WorkerProgress& worker : progress) {
      if (worker.failures.load(std::memory_order_relaxed) >= max_failures_) {
        return absl::InternalError(
            "There was at least one mismatch during execution.");
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace xls::internal
