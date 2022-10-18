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

#include "xls/common/logging/log_message.h"

#include <array>
#include <atomic>
#include <streambuf>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_entry.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/strerror.h"
#include "xls/common/symbolized_stacktrace.h"

namespace xls {
namespace logging_internal {
namespace {

// `global_sinks` holds globally registered `LogSink`s.
ABSL_CONST_INIT absl::Mutex global_sinks_mutex(absl::kConstInit);
ABSL_CONST_INIT std::vector<LogSink*>* global_sinks ABSL_GUARDED_BY(
    global_sinks_mutex) ABSL_PT_GUARDED_BY(global_sinks_mutex) = nullptr;

// `sink_send_mutex` protects against concurrent calls from the logging library
// to any `LogSink::Send()`.
ABSL_CONST_INIT absl::Mutex sink_send_mutex
    ABSL_ACQUIRED_AFTER(global_sinks_mutex)(absl::kConstInit);

// Have we already seen a fatal message?
std::atomic_flag seen_fatal = ATOMIC_FLAG_INIT;

// Copies into `dst` as many bytes of `src` as will fit, then truncates the
// copied bytes from the front of `dst` and returns the number of bytes written.
size_t AppendTruncated(std::string_view src, absl::Span<char>* dst) {
  if (src.size() > dst->size()) src = src.substr(0, dst->size());
  memcpy(dst->data(), src.data(), src.size());
  dst->remove_prefix(src.size());
  return src.size();
}

// Copy of first FATAL log message so that we can print it out again
// after all the stack traces.
ABSL_CONST_INIT absl::Time fatal_time;
ABSL_CONST_INIT std::array<char, 512> fatal_message{{0}};

// A write-only `std::streambuf` that writes into an `absl::Span<char>`.
class SpanStreambuf : public std::streambuf {
 public:
  explicit SpanStreambuf(absl::Span<char> buf) {
    setp(buf.data(), buf.data() + buf.size());
  }

  SpanStreambuf(SpanStreambuf&&) = default;
  SpanStreambuf& operator=(SpanStreambuf&&) = default;

  std::string_view data() const {
    return std::string_view(pbase(), pptr() - pbase());
  }

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    n = std::min<std::streamsize>(n, epptr() - pptr());
    memcpy(pptr(), s, n);
    pbump(n);
    return n;
  }
};

// Returns a mutable reference to a thread-local variable that should be true if
// a `LogSink::Send()` is currently being invoked on this thread.
inline bool& ThreadIsLogging() {
  static thread_local bool thread_is_logging = false;
  return thread_is_logging;
}

}  // namespace

struct LogMessage::LogMessageData {
  LogMessageData(const char* file, int line, absl::LogSeverity severity,
                 absl::Time timestamp);
  LogMessageData(const LogMessageData&) = delete;
  LogMessageData& operator=(const LogMessageData&) = delete;

  // `LogEntry` sent to `LogSink`s; contains metadata.
  LogEntry entry;

  // false => data has not been flushed
  bool has_been_flushed;
  // true => this was first fatal msg
  bool first_fatal;
  // true => all failures should be quiet
  bool fail_quietly;
  // true => PLOG was requested
  bool is_perror;

  // Extra `LogSink`s to log to, in addition to `global_sinks`.
  absl::InlinedVector<LogSink*, 16> extra_sinks;
  // If true, log to `extra_sinks` but not to `global_sinks` or hardcoded
  // non-sink targets (e.g. stderr, log files).
  bool extra_sinks_only;

  // A formatted string message is built in `string_buf`.
  std::array<char, BufferSize()> string_buf;
  // `prefix` is the prefix of `string_buf` containing the log prefix, an
  // automatically generated string containing metadata such as filename, line
  // number, timestamp, etc. about the logged message.  It may be empty if the
  // prefix has been disabled, and in any case it will not be nul-terminated
  // since the message itself follows directly.
  std::string_view prefix;
  // `message` is the prefix of `string_buf` containing the log prefix (if any)
  // and the logged message.  It always ends in a newline and always points into
  // a buffer that contains a nul-terminator in the following byte.
  std::string_view message;

  // A `std::streambuf` that stores into `string_buf`.
  SpanStreambuf streambuf;
};

LogMessage::LogMessageData::LogMessageData(const char* file, int line,
                                           absl::LogSeverity severity,
                                           absl::Time timestamp)
    : entry(file, line, severity, timestamp),
      extra_sinks_only(false),
      streambuf(absl::MakeSpan(string_buf)) {}

LogMessage::LogMessage(const char* file, int line, absl::LogSeverity severity)
    : data_(
          std::make_unique<LogMessageData>(file, line, severity, absl::Now())),
      stream_(&data_->streambuf) {
  bool first_fatal = false;
  if (severity == absl::LogSeverity::kFatal) {
    // Exactly one LOG(FATAL) message is responsible for aborting the process,
    // even if multiple threads LOG(FATAL) concurrently.
    first_fatal = !seen_fatal.test_and_set(std::memory_order_relaxed);
  }

  data_->first_fatal = first_fatal;
  data_->has_been_flushed = false;
  data_->is_perror = false;
  data_->fail_quietly = false;

  // Legacy defaults for LOG's ostream:
  stream_.setf(std::ios_base::showbase | std::ios_base::boolalpha);
  // `fill('0')` is omitted here because its effects are very different without
  // structured logging.

  // This logs a backtrace even if the location is subsequently changed using
  // AtLocation.  This quirk, and the behavior when AtLocation is called twice,
  // are fixable but probably not worth fixing.
  LogBacktraceIfNeeded();
}

LogMessage::~LogMessage() { Flush(); }

LogMessage& LogMessage::WithCheckFailureMessage(std::string_view msg) {
  stream() << "Check failed: " << msg << " ";
  return *this;
}

LogMessage& LogMessage::AtLocation(std::string_view file, int line) {
  data_->entry.set_source_filename(file);
  data_->entry.set_source_line(line);
  LogBacktraceIfNeeded();
  return *this;
}

LogMessage& LogMessage::NoPrefix() {
  data_->entry.set_prefix(false);
  return *this;
}

LogMessage& LogMessage::WithPerror() {
  data_->is_perror = true;
  return *this;
}

LogMessage& LogMessage::WithVerbosity(int verbose_level) {
  if (verbose_level == LogEntry::kNoVerboseLevel) {
    data_->entry.set_verbosity(LogEntry::kNoVerboseLevel);
  } else {
    data_->entry.set_verbosity(std::max(0, verbose_level));
  }
  return *this;
}

LogMessage& LogMessage::ToSinkAlso(LogSink* sink) {
  if (sink != nullptr) {
    data_->extra_sinks.push_back(sink);
  }
  return *this;
}

LogMessage& LogMessage::ToSinkOnly(LogSink* sink) {
  data_->extra_sinks.clear();
  if (sink != nullptr) {
    data_->extra_sinks.push_back(sink);
  }
  data_->extra_sinks_only = true;
  return *this;
}

void LogMessage::Flush() {
  if (data_->has_been_flushed ||
      data_->entry.log_severity() <
          static_cast<absl::LogSeverity>(absl::GetFlag(FLAGS_minloglevel)))
    return;

  if (data_->is_perror) {
    stream() << ": " << Strerror(errno_saver_()) << " [" << errno_saver_()
             << "]";
  }

  data_->entry.set_text_message(data_->streambuf.data());

  if (ThreadIsLogging()) {
    // In the case of recursive logging, just dump the message to stderr.
    if (!data_->extra_sinks_only) {
      size_t written = fwrite(data_->streambuf.data().data(), 1,
                              data_->streambuf.data().size(), stderr);
      // Note: recursive logging where fwrite to stderr fails we silently
      // ignore.
      (void)written;
    }
    return;
  }

  ThreadIsLogging() = true;
  SendToLog();
  ThreadIsLogging() = false;

  // Note that this message is now safely logged.  If we're asked to flush
  // again, as a result of destruction, say, we'll do nothing on future calls.
  data_->has_been_flushed = true;
}

void LogMessage::LogToSinks() const
    ABSL_LOCKS_EXCLUDED(global_sinks_mutex,
                        sink_send_mutex) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (!data_->extra_sinks_only) global_sinks_mutex.ReaderLock();
  if (!data_->extra_sinks.empty() ||
      (!data_->extra_sinks_only && global_sinks && !global_sinks->empty())) {
    {
      absl::MutexLock send_sink_lock(&sink_send_mutex);
      for (LogSink* sink : data_->extra_sinks) {
        sink->Send(data_->entry);
      }
      if (!data_->extra_sinks_only && global_sinks) {
        for (LogSink* sink : *global_sinks) {
          sink->Send(data_->entry);
        }
      }
    }
    for (LogSink* sink : data_->extra_sinks) {
      sink->WaitTillSent();
    }
    if (!data_->extra_sinks_only && global_sinks) {
      for (LogSink* sink : *global_sinks) {
        sink->WaitTillSent();
      }
    }
  }
  if (!data_->extra_sinks_only) global_sinks_mutex.ReaderUnlock();
}

void LogMessage::Fail() {
  // Pre-output stack trace now (similarly to FailureSignalHandler):
  // sometimes it's too difficult to recover the full stack trace from
  // within the abnormal termination signal handler into which we are
  // about to jump, so we dump the trace before that.  We write a
  // stack trace, for the current thread ONLY to stderr.  That way,
  // even if nothing else works, at least we get a basic stack.  This
  // should be async-termination-safe.
  std::string stack_trace =
      GetSymbolizedStackTraceAsString(/*max_depth=*/50, /*skip_count=*/1);
  fprintf(stderr, "%s", stack_trace.c_str());
  FailWithoutStackTrace();
}

#if __ELF__
extern "C" void __gcov_flush() ABSL_ATTRIBUTE_WEAK;
#endif

void LogMessage::FailWithoutStackTrace() {
#if defined _DEBUG && defined COMPILER_MSVC
  // When debugging on windows, avoid the obnoxious dialog.
  __debugbreak();
#endif

#if __ELF__
  // Flush coverage if we are in coverage mode.
  if (&__gcov_flush != nullptr) {
    __gcov_flush();
  }
#endif

  abort();
}

void LogMessage::FailQuietly() {
  // _exit. Calling abort() would trigger all sorts of death signal handlers
  // and a detailed stack trace. Calling exit() would trigger the onexit
  // handlers, including the heap-leak checker, which is guaranteed to fail in
  // this case: we probably just new'ed the std::string that we logged.
  // Anyway, if you're calling Fail or FailQuietly, you're trying to bail out
  // of the program quickly, and it doesn't make much sense for FailQuietly to
  // offer different guarantees about exit behavior than Fail does. (And as a
  // consequence for QCHECK and CHECK to offer different exit behaviors)
  _exit(1);
}

void LogMessage::SetFailQuietly() { data_->fail_quietly = true; }

namespace {
// We evaluate `FLAGS_log_backtrace_at` as a hash comparison to avoid having to
// hold a mutex or make a copy (to access the value of a string-typed flag) in
// this very hot codepath.
ABSL_CONST_INIT std::atomic<size_t> log_backtrace_at_hash{0};

size_t HashSiteForLogBacktraceAt(std::string_view file, int line) {
  using HashTuple = std::tuple<std::string_view, int>;
  return absl::Hash<HashTuple>()(HashTuple(file, line));
}
}  // namespace

void LogMessage::LogBacktraceIfNeeded() {
  const size_t flag_hash =
      log_backtrace_at_hash.load(std::memory_order_relaxed);
  if (!flag_hash) return;
  const size_t site_hash = HashSiteForLogBacktraceAt(
      data_->entry.source_basename(), data_->entry.source_line());
  if (site_hash != flag_hash) return;
  stream_ << " (stacktrace:\n";
  stream_ << GetSymbolizedStackTraceAsString(/*max_depth=*/50,
                                             /*skip_count=*/1);
  stream_ << ") ";
}

void LogMessage::PrepareToDieIfFatal() {
  if (data_->entry.log_severity() != absl::LogSeverity::kFatal) {
    return;
  }
  // If we log a FATAL message, flush all the log destinations, then toss
  // a signal for others to catch. We leave the logs in a state that
  // someone else can use them (as long as they flush afterwards)
  if (data_->first_fatal) {
    // Store shortened fatal message for other logs and GWQ status.
    std::string_view message = data_->streambuf.data();
    auto fatal_message_remaining = absl::MakeSpan(fatal_message);
    // We may need to write a newline and nul-terminator at the end of the
    // copied message (i.e. if it was truncated).  Rather than worry about
    // whether those should overwrite the end of the string (if the buffer is
    // full) or be appended, we avoid writing into the last two bytes so we
    // always have space to append.
    fatal_message_remaining.remove_suffix(2);
    size_t chars_written = AppendTruncated(message, &fatal_message_remaining);
    // Append a '\n' unless the message already ends with one.
    if (!chars_written || fatal_message[chars_written - 1] != '\n')
      fatal_message[chars_written++] = '\n';
    fatal_message[chars_written] = '\0';
    fatal_time = data_->entry.timestamp();
  }
}

void LogMessage::DieIfFatal() {
  if (data_->entry.log_severity() != absl::LogSeverity::kFatal) {
    return;
  }
  if (data_->fail_quietly) {
    FailQuietly();
  } else {
    if (!data_->extra_sinks_only) {
      std::string message = "*** Check failure stack trace: ***\n";
      ssize_t retcode = write(STDERR_FILENO, message.c_str(), message.length());
      (void)retcode;
      std::string stack_trace =
          GetSymbolizedStackTraceAsString(/*max_depth=*/50,
                                          /*skip_count=*/1);
      fprintf(stderr, "%s", stack_trace.c_str());
      if (absl::StderrThreshold() > absl::LogSeverity::kInfo) {
        message.append(stack_trace);
      }
    }
    // Clear anti-recursion marker to allow FailureSignalHandler to
    // log during its unsafe phase.
    ThreadIsLogging() = false;
    FailWithoutStackTrace();
  }
}

void LogMessage::SendToLog() {
  if (!data_->extra_sinks_only) {
    const std::string message = absl::StrCat(data_->entry.FormatPrefix(),
                                             data_->streambuf.data(), "\n");
    const std::string formatted = absl::StrCat(
        data_->entry.source_basename(), ":", data_->entry.source_line(), " ",
        data_->streambuf.data(), "\n");

    if (absl::StderrThreshold() == absl::LogSeverityAtLeast::kInfo ||
        (data_->entry.log_severity() >= absl::StderrThreshold())) {
      // TODO(xls-team): extract the below to a LogSink and remove from here.
      fwrite(message.data(), sizeof(char), message.size(), stderr);
#if _WIN32
      // C99 requires stderr to not be fully-buffered by default (7.19.3.7), but
      // MS CRT buffers it anyway, so we must `fflush` to ensure the string hits
      // the console/file before the program dies (and takes the libc buffers
      // with it).
      // https://docs.microsoft.com/en-us/cpp/c-runtime-library/stream-i-o
      if (data_->entry.log_severity() >= absl::LogSeverity::kWarning) {
        fflush(stderr);
      }
#endif  // _WIN32
    }
  }
  PrepareToDieIfFatal();
  // Also log to all registered sinks, even if OnlyLogToStderr() is set.
  LogToSinks();
  DieIfFatal();
}

template LogMessage& LogMessage::operator<<(const char& v);
template LogMessage& LogMessage::operator<<(const signed char& v);
template LogMessage& LogMessage::operator<<(const unsigned char& v);
template LogMessage& LogMessage::operator<<(const short& v);           // NOLINT
template LogMessage& LogMessage::operator<<(const unsigned short& v);  // NOLINT
template LogMessage& LogMessage::operator<<(const int& v);
template LogMessage& LogMessage::operator<<(const unsigned int& v);
template LogMessage& LogMessage::operator<<(const long& v);           // NOLINT
template LogMessage& LogMessage::operator<<(const unsigned long& v);  // NOLINT
template LogMessage& LogMessage::operator<<(const long long& v);      // NOLINT
template LogMessage& LogMessage::operator<<(
    const unsigned long long& v);  // NOLINT
template LogMessage& LogMessage::operator<<(void* const& v);
template LogMessage& LogMessage::operator<<(const void* const& v);
template LogMessage& LogMessage::operator<<(const float& v);
template LogMessage& LogMessage::operator<<(const double& v);
template LogMessage& LogMessage::operator<<(const bool& v);
template LogMessage& LogMessage::operator<<(const std::string& v);
template LogMessage& LogMessage::operator<<(const std::string_view& v);
LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, absl::LogSeverity::kFatal) {}

LogMessageFatal::LogMessageFatal(const char* file, int line,
                                 std::string_view failure_msg)
    : LogMessage(file, line, absl::LogSeverity::kFatal) {
  WithCheckFailureMessage(failure_msg);
}

// ABSL_ATTRIBUTE_NORETURN doesn't seem to work on destructors with msvc, so
// disable msvc's warning about the d'tor never returning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4722)
#endif
LogMessageFatal::~LogMessageFatal() {
  Flush();
  LogMessage::FailWithoutStackTrace();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

LogMessageQuietlyFatal::LogMessageQuietlyFatal(const char* file, int line)
    : LogMessage(file, line, absl::LogSeverity::kFatal) {
  SetFailQuietly();
}

LogMessageQuietlyFatal::LogMessageQuietlyFatal(const char* file, int line,
                                               std::string_view failure_msg)
    : LogMessage(file, line, absl::LogSeverity::kFatal) {
  SetFailQuietly();
  WithCheckFailureMessage(failure_msg);
}

LogMessageQuietlyFatal::~LogMessageQuietlyFatal() {
  Flush();
  LogMessage::FailQuietly();
}

}  // namespace logging_internal

void AddLogSink(LogSink* sink)
    ABSL_LOCKS_EXCLUDED(logging_internal::global_sinks_mutex) {
  absl::MutexLock global_sinks_lock(&logging_internal::global_sinks_mutex);
  if (!logging_internal::global_sinks)
    logging_internal::global_sinks = new std::vector<LogSink*>();
  logging_internal::global_sinks->push_back(sink);
}

void RemoveLogSink(LogSink* sink)
    ABSL_LOCKS_EXCLUDED(logging_internal::global_sinks_mutex) {
  absl::MutexLock global_sinks_lock(&logging_internal::global_sinks_mutex);
  if (!logging_internal::global_sinks) return;
  for (auto iter = logging_internal::global_sinks->begin();
       iter != logging_internal::global_sinks->end(); ++iter) {
    if (*iter == sink) {
      logging_internal::global_sinks->erase(iter);
      return;
    }
  }
}

}  // namespace xls
