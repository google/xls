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

#include "xls/common/status/status_builder.h"

#include <cstdio>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "absl/base/log_severity.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "xls/common/logging/logging.h"
#include "xls/common/source_location.h"
#include "xls/common/symbolized_stacktrace.h"

namespace xabsl {

StatusBuilder::Rep::Rep(const Rep& r)
    : logging_mode(r.logging_mode),
      log_severity(r.log_severity),
      verbose_level(r.verbose_level),
      n(r.n),
      period(r.period),
      stream(r.stream.str()),
      should_log_stack_trace(r.should_log_stack_trace),
      message_join_style(r.message_join_style),
      sink(r.sink) {}

absl::Status StatusBuilder::JoinMessageToStatus(absl::Status s,
                                                std::string_view msg,
                                                MessageJoinStyle style) {
  if (msg.empty()) {
    return s;
  }
  if (style == MessageJoinStyle::kAnnotate) {
    return AnnotateStatus(s, msg);
  }
  std::string new_msg = style == MessageJoinStyle::kPrepend
                            ? absl::StrCat(msg, s.message())
                            : absl::StrCat(s.message(), msg);
  absl::Status result = WithMessage(s, new_msg);
  SetStatusCode(s.code(), &result);
  return result;
}

void StatusBuilder::ConditionallyLog(const absl::Status& status) const {
  if (rep_->logging_mode == Rep::LoggingMode::kDisabled) {
    return;
  }

  absl::LogSeverity severity = rep_->log_severity;
  switch (rep_->logging_mode) {
    case Rep::LoggingMode::kDisabled:
    case Rep::LoggingMode::kLog:
      break;
    case Rep::LoggingMode::kVLog: {
      // Combine these into a single struct so that we only have one atomic
      // access on each pass through the function (instead of one for the map
      // and one for the mutex).
      struct LogSites {
        absl::Mutex mutex;
        // NOLINTNEXTLINE(abseil-no-internal-dependencies)
        std::unordered_map<const void*, absl::log_internal::VLogSite>
            sites_by_file ABSL_GUARDED_BY(mutex);
      };
      static auto* vlog_sites = new LogSites();

      vlog_sites->mutex.Lock();
      // This assumes that loc_.file_name() is a compile time constant in order
      // to satisfy the lifetime constraints imposed by VLogSite. The
      // constructors of SourceLocation guarantee that for us.
      auto [iter, unused] = vlog_sites->sites_by_file.try_emplace(
          loc_.file_name(), loc_.file_name());
      auto& site = iter->second;
      vlog_sites->mutex.Unlock();

      if (!site.IsEnabled(rep_->verbose_level)) {
        return;
      }

      severity = absl::LogSeverity::kInfo;
      break;
    }
    case Rep::LoggingMode::kLogEveryN: {
      struct LogSites {
        absl::Mutex mutex;
        absl::flat_hash_map<std::pair<const void*, uint>, uint>
            counts_by_file_and_line ABSL_GUARDED_BY(mutex);
      };
      static auto* log_every_n_sites = new LogSites();

      log_every_n_sites->mutex.Lock();
      const uint count =
          log_every_n_sites
              ->counts_by_file_and_line[{loc_.file_name(), loc_.line()}]++;
      log_every_n_sites->mutex.Unlock();

      if (count % rep_->n != 0) {
        return;
      }
      break;
    }
    case Rep::LoggingMode::kLogEveryPeriod: {
      struct LogSites {
        absl::Mutex mutex;
        absl::flat_hash_map<std::pair<const void*, uint>, absl::Time>
            next_log_by_file_and_line ABSL_GUARDED_BY(mutex);
      };
      static auto* log_every_sites = new LogSites();

      const auto now = absl::Now();
      absl::MutexLock lock(&log_every_sites->mutex);
      absl::Time& next_log =
          log_every_sites
              ->next_log_by_file_and_line[{loc_.file_name(), loc_.line()}];
      if (now < next_log) {
        return;
      }
      next_log = now + rep_->period;
      break;
    }
  }

  absl::LogSink* const sink = rep_->sink;
  const std::string maybe_stack_trace =
      rep_->should_log_stack_trace
          ? absl::StrCat("\n", xls::GetSymbolizedStackTraceAsString(
                                   /*max_depth=*/50, /*skip_count=*/1))
          : "";
  const int verbose_level = rep_->logging_mode == Rep::LoggingMode::kVLog
                                ? rep_->verbose_level
                                : absl::LogEntry::kNoVerboseLevel;
  if (sink) {
    LOG(LEVEL(severity))
            .AtLocation(loc_.file_name(), loc_.line())
            .ToSinkAlso(sink)
            .WithVerbosity(verbose_level)
        << status << maybe_stack_trace;
  } else {
    // sink == nullptr indicates not to call ToSinkAlso(), which dies if sink is
    // nullptr. Unfortunately, this means we reproduce the above macro call.
    LOG(LEVEL(severity))
            .AtLocation(loc_.file_name(), loc_.line())
            .WithVerbosity(verbose_level)
        << status << maybe_stack_trace;
  }
}

void StatusBuilder::SetStatusCode(absl::StatusCode canonical_code,
                                  absl::Status* status) {
  if (status->code() == canonical_code) {
    return;
  }
  absl::Status new_status(canonical_code, status->message());
  CopyPayloads(*status, &new_status);
  using std::swap;
  swap(*status, new_status);
}

void StatusBuilder::CopyPayloads(const absl::Status& src, absl::Status* dst) {
  src.ForEachPayload([&](std::string_view type_url, absl::Cord payload) {
    dst->SetPayload(type_url, payload);
  });
}

absl::Status StatusBuilder::WithMessage(const absl::Status& status,
                                        std::string_view msg) {
  // Unfortunately since we can't easily strip the source-location off of this
  // new status the backtrace can end up with a lot of copies of this line at
  // the beginning. We manually try to trim them out but we can't actually
  // remove the first one.
  auto ret = absl::Status(status.code(), msg);
  std::optional<SourceLocation> first =
      StatusBuilder::GetSourceLocations(ret).empty()
          ? std::nullopt
          : std::make_optional<SourceLocation>(
                StatusBuilder::GetSourceLocations(ret).front());
  bool first_non_duplicate = false;
  for (const SourceLocation& sl : StatusBuilder::GetSourceLocations(status)) {
    if (!first_non_duplicate && first && first->line() == sl.line() &&
        std::string_view(first->file_name()) ==
            std::string_view(sl.file_name())) {
      continue;
    }
    first_non_duplicate = true;
    StatusBuilder::AddSourceLocation(ret, sl);
  }
  CopyPayloads(status, &ret);
  return ret;
}

absl::Status StatusBuilder::AnnotateStatus(const absl::Status& s,
                                           std::string_view msg) {
  if (s.ok() || msg.empty()) {
    return s;
  }

  std::string_view new_msg = msg;
  std::string annotated;
  if (!s.message().empty()) {
    absl::StrAppend(&annotated, s.message(), "; ", msg);
    new_msg = annotated;
  }
  absl::Status result = WithMessage(s, new_msg);
  SetStatusCode(s.code(), &result);
  return result;
}

absl::Status StatusBuilder::CreateStatusAndConditionallyLog() && {
  absl::Status result = JoinMessageToStatus(
      std::move(status_), rep_->stream.str(), rep_->message_join_style);
  ConditionallyLog(result);
  StatusBuilder::AddSourceLocation(result, loc_);

  // We consumed the status above, we set it to some error just to prevent
  // people relying on it become OK or something.
  status_ = absl::UnknownError("");
  rep_ = nullptr;
  return result;
}

std::ostream& operator<<(std::ostream& os, const StatusBuilder& builder) {
  return os << static_cast<absl::Status>(builder);
}

std::ostream& operator<<(std::ostream& os, StatusBuilder&& builder) {
  return os << static_cast<absl::Status>(std::move(builder));
}

StatusBuilder AbortedErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kAborted, location);
}

StatusBuilder AlreadyExistsErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kAlreadyExists, location);
}

StatusBuilder CancelledErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kCancelled, location);
}

StatusBuilder DataLossErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kDataLoss, location);
}

StatusBuilder DeadlineExceededErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kDeadlineExceeded, location);
}

StatusBuilder FailedPreconditionErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kFailedPrecondition, location);
}

StatusBuilder InternalErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kInternal, location);
}

StatusBuilder InvalidArgumentErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kInvalidArgument, location);
}

StatusBuilder NotFoundErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kNotFound, location);
}

StatusBuilder OutOfRangeErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kOutOfRange, location);
}

StatusBuilder PermissionDeniedErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kPermissionDenied, location);
}

StatusBuilder UnauthenticatedErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kUnauthenticated, location);
}

StatusBuilder ResourceExhaustedErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kResourceExhausted, location);
}

StatusBuilder UnavailableErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kUnavailable, location);
}

StatusBuilder UnimplementedErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kUnimplemented, location);
}

StatusBuilder UnknownErrorBuilder(xabsl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kUnknown, location);
}

}  // namespace xabsl
