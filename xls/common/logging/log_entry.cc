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

#include "xls/common/logging/log_entry.h"

#include <time.h>

#include "absl/base/internal/sysinfo.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/log_flags.h"

namespace xls {
namespace {

// GetCachedTID() caches the thread ID in thread-local storage (which is a
// userspace construct) to avoid unnecessary system calls. Without this caching,
// it can take roughly 98ns, while it takes roughly 1ns with this caching.
pid_t GetCachedTID() {
  static thread_local pid_t thread_id = absl::base_internal::GetTID();
  return thread_id;
}

std::string_view Basename(std::string_view filepath) {
#ifdef _WIN32
  size_t path = filepath.find_last_of("/\\");
#else
  size_t path = filepath.find_last_of('/');
#endif
  if (path != filepath.npos) filepath.remove_prefix(path + 1);
  return filepath;
}

// This is the size of the portion of the log prefix before the filename:
constexpr size_t kFixedPrefixLen =
    sizeof("SMMDD HH:MM:SS.NNNNNN TTTTTTT ") - sizeof("");

char* AppendTwoDigit(char* buf, uint32_t v) {
  buf[1] = v % 10 + '0';
  buf[0] = v / 10 + '0';
  return buf + 2;
}

// Append an ascii representation of "v" to "*out", left-padding with
// repetitions of "fill" if the representation is less than "width" characters
// long.
// REQUIRES: "out[]" must have sufficient space to hold the ascii string.
char* AppendUint(char* out, uint32_t v, char fill, int width) {
  char buf[32];  // Plenty to hold ascii representation of an int
  int p = 32;
  do {
    buf[--p] = (v % 10) + '0';
    v /= 10;
  } while (v > 0);
  while (p > 32 - width) {
    buf[--p] = fill;
  }
  // Copy formatted data in 'buf' to 'out' and return pointer to next
  // position to use in out
  const size_t n = 32 - p;
  memcpy(out, buf + p, n);
  return out + n;
}

}  // namespace

LogEntry::LogEntry(std::string_view full_filename, int line,
                   absl::LogSeverity severity, absl::Time timestamp)
    : full_filename_(full_filename),
      base_filename_(Basename(full_filename)),
      line_(line),
      prefix_(true),
      severity_(absl::NormalizeLogSeverity(severity)),
      verbose_level_(kNoVerboseLevel),
      timestamp_(timestamp),
      tid_(GetCachedTID()),
      text_message_("") {
  GenerateTimestampAsTm();
}

std::string LogEntry::ToString() const {
  return absl::StrCat(FormatPrefix(), text_message_);
}

void LogEntry::set_source_filename(std::string_view filename) {
  full_filename_ = filename;
  base_filename_ = Basename(full_filename_);
}

void LogEntry::GenerateTimestampAsTm() {
#if defined(_WIN32)
  time_t time_secs = absl::ToTimeT(timestamp_);
  localtime_s(&timestamp_as_tm_, &time_secs);
#else
  time_t time_secs = absl::ToTimeT(timestamp_);
  localtime_r(&time_secs, &timestamp_as_tm_);
#endif
}

void LogEntry::AppendSeverityTimeAndThreadId(std::string* out) const {
  // Append something like 'I0513 17:35:46.294773   27319 ' to *out
  char buf[kFixedPrefixLen];
  char* p = buf;
  *p++ = absl::LogSeverityName(severity_)[0];
  p = AppendTwoDigit(p, timestamp_as_tm_.tm_mon + 1);
  p = AppendTwoDigit(p, timestamp_as_tm_.tm_mday);
  *p++ = ' ';
  p = AppendTwoDigit(p, timestamp_as_tm_.tm_hour);
  *p++ = ':';
  p = AppendTwoDigit(p, timestamp_as_tm_.tm_min);
  *p++ = ':';
  p = AppendTwoDigit(p, timestamp_as_tm_.tm_sec);
  *p++ = '.';
  int usecs = absl::ToTimeval(timestamp_).tv_usec;
  // Three lines below are equivalent but make logging ~3% faster overall than:
  //    p = AppendUint(p, usecs, '0', 6);
  p = AppendTwoDigit(p, usecs / 10000);
  p = AppendTwoDigit(p, (usecs / 100) % 100);
  p = AppendTwoDigit(p, usecs % 100);
  *p++ = ' ';
  p = AppendUint(p, tid_, ' ', 7);
  *p++ = ' ';
  assert(p - buf <= sizeof(buf));
  out->append(buf, p - buf);
}

std::string LogEntry::FormatPrefix() const {
  std::string prefix;
  if (!absl::GetFlag(FLAGS_log_prefix) || !prefix_ || line_ == -1)
    return prefix;
  // Generate a prefix like:
  // 'I0513 17:35:46.294773   27319 logging_unittest.cc:147] '
  prefix.reserve(kFixedPrefixLen + base_filename_.size());

  AppendSeverityTimeAndThreadId(&prefix);
  absl::StrAppend(&prefix, base_filename_);
  // Generate ':<line>] '
  char buf[kFixedPrefixLen];
  char* p = buf;
  *p++ = ':';
  p = AppendUint(p, line_, ' ', 0);
  *p++ = ']';
  *p++ = ' ';
  prefix.append(buf, p - buf);
  // This StrFormat should do the same formatting as the above, but it's
  // likely to run slower and invoke the allocator, so we stick with manual
  // char-banging in optimized builds.
  // n.b.: Check if the basename is this file to prevent recursive calls.
  assert(base_filename_ == "logging.cc" ||
         prefix ==
             absl::StrFormat("%c%02d%02d %02d:%02d:%02d.%06d %7u %s:%d] ",
                             absl::LogSeverityName(severity_)[0],
                             timestamp_as_tm_.tm_mon + 1,
                             timestamp_as_tm_.tm_mday, timestamp_as_tm_.tm_hour,
                             timestamp_as_tm_.tm_min, timestamp_as_tm_.tm_sec,
                             absl::ToTimeval(timestamp_).tv_usec, tid_,
                             base_filename_, line_));
  return prefix;
}

}  // namespace xls
