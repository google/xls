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

#ifndef XLS_COMMON_LOGGING_LOG_ENTRY_H_
#define XLS_COMMON_LOGGING_LOG_ENTRY_H_

#include "absl/base/log_severity.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace xls {
namespace logging_internal {
class LogMessage;
}

// Represents a log record as passed to `LogSink::Send`.
// Data returned by pointer or by reference must be copied if they are needed
// after the lifetime of the `LogEntry`.
class LogEntry {
 public:
  // For non-verbose log entries, `verbosity()` returns `kNoVerboseLevel`.
  static constexpr int kNoVerboseLevel = -1;

  LogEntry(std::string_view full_filename, int line,
           absl::LogSeverity severity, absl::Time timestamp);

  // Format this `LogEntry` as it should appear in a log file.  Most
  // `LogSink::Send` overrides should use this method to obtain a log string.
  std::string ToString() const;

  // Source file and line where the log message occurred.
  // Take special care not to dereference the pointers returned by
  // source_filename() and source_basename() after the lifetime of the
  // `LogEntry`. This will usually work, because these are usually backed by a
  // statically allocated char array obtained from the `__FILE__` macro, but
  // it is nevertheless incorrect and will be broken by statements like
  // `LOG(INFO).AtLocation(...)` (see above).  If you need the data later, you
  // must copy it.
  std::string_view source_filename() const { return full_filename_; }
  std::string_view source_basename() const { return base_filename_; }
  // This sets both source_filename and base_filename.
  void set_source_filename(std::string_view filename);
  int source_line() const { return line_; }
  void set_source_line(int source_line) { line_ = source_line; }

  // True unless cleared by LOG(...).NoPrefix(), which indicates suppression of
  // the line prefix containing metadata like file, line, timestamp, etc.
  bool prefix() const { return prefix_; }
  void set_prefix(bool prefix) { prefix_ = prefix; }

  // Severity.
  absl::LogSeverity log_severity() const { return severity_; }
  void set_log_severity(absl::LogSeverity log_severity) {
    severity_ = absl::NormalizeLogSeverity(log_severity);
  }

  // Verbosity (returns kNoVerboseLevel for non-verbose log entries).
  int verbosity() const { return verbose_level_; }
  void set_verbosity(int verbosity) { verbose_level_ = verbosity; }

  // Time that the message occurred.
  absl::Time timestamp() const { return timestamp_; }
  void set_timestamp(absl::Time timestamp) {
    timestamp_ = timestamp;
    GenerateTimestampAsTm();
  }
  // Breakdown of `timestamp()` in the local time zone.
  // Prefer using `timestamp()` unless you specifically need this breakdown
  // to do ASCII formatting, etc.
  const struct tm& timestamp_as_tm() const { return timestamp_as_tm_; }

#ifdef _WIN32
  uint32_t tid() const { return tid_; }
  void set_tid(uint32_t tid) { tid_ = tid; }
#else
  pid_t tid() const { return tid_; }
  void set_tid(pid_t tid) { tid_ = tid; }
#endif

  // Text-formatted version of the log message.
  // This does not include the prefix or a trailing newline; consider
  // `ToString()` if you require the prefix.
  std::string_view text_message() const { return text_message_; }
  void set_text_message(std::string_view text_message) {
    text_message_ = text_message;
  }

 private:
  void GenerateTimestampAsTm();
  void AppendSeverityTimeAndThreadId(std::string* out) const;
  std::string FormatPrefix() const;

  std::string_view full_filename_;
  std::string_view base_filename_;
  int line_;
  bool prefix_;
  absl::LogSeverity severity_;
  int verbose_level_;  // >=0 for `VLOG`, etc.; otherwise `kNoVerboseLevel`.
  absl::Time timestamp_;
  struct tm timestamp_as_tm_;
#ifdef _WIN32
  uint32_t tid_;
#else
  pid_t tid_;
#endif
  std::string_view text_message_;

  friend class ::xls::logging_internal::LogMessage;
};

}  // namespace xls

#endif  // XLS_COMMON_LOGGING_LOG_ENTRY_H_
