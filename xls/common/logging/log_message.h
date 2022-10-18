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

#ifndef XLS_COMMON_LOGGING_LOG_MESSAGE_H_
#define XLS_COMMON_LOGGING_LOG_MESSAGE_H_

#include <memory>
#include <ostream>

#include "absl/base/attributes.h"
#include "absl/base/log_severity.h"
#include "absl/strings/string_view.h"
#include "xls/common/logging/errno_saver.h"
#include "xls/common/logging/log_sink.h"
#include "xls/common/logging/null_guard.h"
#include "xls/common/source_location.h"

namespace xls {
namespace logging_internal {

// This class more or less represents a particular log message.  You create a
// temporary instance of `LogMessage` and then stream values to it.  At the end
// of the statement, it goes out of scope and `~LogMessage` writes the message
// out as appropriate.
// Heap-allocation of `LogMessage` is unsupported.  Construction outside of a
// `XLS_LOG` macro is strongly discouraged.
class LogMessage {
 public:
  // The size of the buffer in which structured log data is stored, and also of
  // the separate buffer in which text data is stored.
  static constexpr size_t BufferSize() { return 15000; }

  // Used for `XLS_LOG`.
  LogMessage(const char* file, int line,
             absl::LogSeverity severity) ABSL_ATTRIBUTE_COLD;

  LogMessage(const LogMessage&) = delete;
  LogMessage& operator=(const LogMessage&) = delete;

  ~LogMessage() ABSL_ATTRIBUTE_COLD;

  // Streams `"Check failed: " << msg << " "` into the buffer.  We avoid binary
  // bloat by keeping a single copy of the literals in logging.cc (vs.
  // preprocessor-concatenating them into every expansion site's `#msg`).  We
  // also facilitate mutator methods on `XLS_CHECK` XLS_and `CHECK_EQ` by doing
  // our streaming inside a chainable method call instead of explicitly in the
  // macro expansion.  Streaming decays the type of the expression from
  // `LogMessage&` to `std::ostream&`.
  LogMessage& WithCheckFailureMessage(std::string_view msg);

  // Overrides the location inferred from the callsite.  The string pointed to
  // by `file` must be valid until the end of the statement.
  LogMessage& AtLocation(std::string_view file, int line);
  // `loc` doesn't default to `SourceLocation::current()` here since the
  // callsite is already the default location for `LOG` statements.
  LogMessage& AtLocation(xabsl::SourceLocation loc) {
    return AtLocation(loc.file_name(), loc.line());
  }

  // Omits the prefix from this line.  The prefix includes metadata about the
  // logged data such as source code location and timestamp.
  LogMessage& NoPrefix();
  // Appends to the logged message a colon, a space, a textual description of
  // the current value of `errno` (as by strerror(3)), and the numerical value
  // of `errno`.
  LogMessage& WithPerror();
  // Sets the verbosity field of the logged message as if it was logged by
  // `XLS_VLOG(verbose_level)`.  Unlike `XLS_VLOG`, this method does not affect
  // evaluation of the statement when the specified `verbose_level` has been
  // disabled.  The only effect is on `LogSink` implementations which make use
  // of the `LogSink::verbosity()` value.  The value `LogEntry::kNoVerboseLevel`
  // can be specified to mark the message not verbose.
  LogMessage& WithVerbosity(int verbose_level);
  // Sends this message to `*sink` in addition to whatever other sinks it would
  // otherwise have been sent to.  `sink` must not be null.
  LogMessage& ToSinkAlso(LogSink* sink);
  // Sends this message to `*sink` and no others.  `sink` must not be null.
  LogMessage& ToSinkOnly(LogSink* sink);

  LogMessage& stream() { return *this; }

  // LogMessage accepts streamed values as if it were an ostream.

  // By-value overloads for small, common types let us overlook common failures
  // to define globals and static data members (i.e. in a .cc file).
  // clang-format off
  // The CUDA toolchain cannot handle these <<<'s:
  LogMessage& operator<<(char v) { return operator<< <char>(v); }
  LogMessage& operator<<(signed char v) { return operator<< <signed char>(v); }
  LogMessage& operator<<(unsigned char v) {
    return operator<< <unsigned char>(v);
  }
  LogMessage& operator<<(signed short v) {  // NOLINT
    return operator<< <signed short>(v);  // NOLINT
  }
  LogMessage& operator<<(signed int v) { return operator<< <signed int>(v); }
  LogMessage& operator<<(signed long v) {  // NOLINT
    return operator<< <signed long>(v);  // NOLINT
  }
  LogMessage& operator<<(signed long long v) {  // NOLINT
    return operator<< <signed long long>(v);  // NOLINT
  }
  LogMessage& operator<<(unsigned short v) {  // NOLINT
    return operator<< <unsigned short>(v);  // NOLINT
  }
  LogMessage& operator<<(unsigned int v) {
    return operator<< <unsigned int>(v);
  }
  LogMessage& operator<<(unsigned long v) {  // NOLINT
    return operator<< <unsigned long>(v);  // NOLINT
  }
  LogMessage& operator<<(unsigned long long v) {  // NOLINT
    return operator<< <unsigned long long>(v);  // NOLINT
  }
  LogMessage& operator<<(void* v) { return operator<< <void*>(v); }
  LogMessage& operator<<(const void* v) { return operator<< <const void*>(v); }
  LogMessage& operator<<(float v) { return operator<< <float>(v); }
  LogMessage& operator<<(double v) { return operator<< <double>(v); }
  LogMessage& operator<<(bool v) { return operator<< <bool>(v); }
  // clang-format on

  // Handle stream manipulators e.g. std::endl.
  LogMessage& operator<<(std::ostream& (*m)(std::ostream& os));
  LogMessage& operator<<(std::ios_base& (*m)(std::ios_base& os));

  // Literal strings.  This allows us to record C string literals as literals in
  // the logging.proto.Value.
  //
  // Allow this overload to be inlined to prevent generating instantiations of
  // this template for every value of `SIZE` encountered in each source code
  // file. That significantly increases linker input sizes. Inlining is cheap
  // because the argument to this overload is almost always a string literal so
  // the call to `strlen` can be replaced at compile time. The overload for
  // `char[]` below should not be inlined. The compiler typically does not have
  // the string at compile time and cannot replace the call to `strlen` so
  // inlining it increases the binary size.
  template <int SIZE>
  LogMessage& operator<<(const char (&buf)[SIZE]);

  // This is prevents non-const `char[]` arrays from looking like literals.
  template <int SIZE>
  LogMessage& operator<<(char (&buf)[SIZE]) ABSL_ATTRIBUTE_NOINLINE;

  // Default: uses `ostream` logging to convert `v` to a string.
  template <typename T>
  LogMessage& operator<<(const T& v) ABSL_ATTRIBUTE_NOINLINE;

  // Note: We explicitly do not support `operator<<` for non-const references
  // because it breaks logging of non-integer bitfield types (i.e., enums).

  // Helper to trick the compiler into passing-in address of a temporary
  // `LogMessage`, e.g.:
  //
  //   MyWrapper(LogMessage(...).self(), ...).my_stream() << ...;
  //
  // The caller must ensure that this `LogMessage` pointer is not used past the
  // lifetime of the temporary object, e.g. by using it only in the wrapping
  // temporary object.
  LogMessage* self() { return this; }

  // Call `abort()` or similar to perform `XLS_LOG(FATAL)` crash.  Writes
  // current stack trace to stderr.
  ABSL_ATTRIBUTE_NORETURN static void Fail();

  // Same as `Fail()`, but without writing out the stack trace.  It is assumed
  // that the caller has already generated and written the trace as appropriate.
  ABSL_ATTRIBUTE_NORETURN static void FailWithoutStackTrace();

 private:
  struct LogMessageData;  // Opaque type containing message state

  void LogToSinks() const;

  void SendToLog();

 public:
  // Similar to `FailWithoutStackTrace()`, but without `abort()`.  Terminates
  // the process with an error exit code.
  ABSL_ATTRIBUTE_NORETURN static void FailQuietly();

 protected:
  // After this is called, failures are done as quiet as possible for this log
  // message.
  void SetFailQuietly();

 private:
  // Checks `FLAGS_log_backtrace_at` and appends a backtrace if appropriate.
  void LogBacktraceIfNeeded();

  // Records some tombstone-type data if the message is fatal, but doesn't die
  // yet.  First we will write to the log.
  void PrepareToDieIfFatal();
  // This one dies if the message is fatal.
  void DieIfFatal();

 protected:
  void Flush();

 private:
  // This should be the first data member so that its initializer captures errno
  // before any other initializers alter it (e.g. with calls to new) and so that
  // no other destructors run afterward an alter it (e.g. with calls to delete).
  ErrnoSaver errno_saver_;

  // We keep the data in a separate struct so that each instance of `LogMessage`
  // uses less stack space.
  std::unique_ptr<LogMessageData> data_;

  std::ostream stream_;
};

// Note: the following is declared `ABSL_ATTRIBUTE_NOINLINE`
template <typename T>
LogMessage& LogMessage::operator<<(const T& v) {
  stream_ << NullGuard<T>().Guard(v);
  return *this;
}
inline LogMessage& LogMessage::operator<<(
    std::ostream& (*m)(std::ostream& os)) {
  stream_ << m;
  return *this;
}
inline LogMessage& LogMessage::operator<<(
    std::ios_base& (*m)(std::ios_base& os)) {
  stream_ << m;
  return *this;
}
template <int SIZE>
LogMessage& LogMessage::operator<<(const char (&buf)[SIZE]) {
  stream_ << buf;
  return *this;
}
// Note: the following is declared `ABSL_ATTRIBUTE_NOINLINE`
template <int SIZE>
LogMessage& LogMessage::operator<<(char (&buf)[SIZE]) {
  stream_ << buf;
  return *this;
}

// We instantiate these specializations in the library's TU to save space in
// other TUs.  Since the template is marked `ABSL_ATTRIBUTE_NOINLINE` we will be
// emitting a function call either way.
extern template LogMessage& LogMessage::operator<<(const char& v);
extern template LogMessage& LogMessage::operator<<(const signed char& v);
extern template LogMessage& LogMessage::operator<<(const unsigned char& v);
extern template LogMessage& LogMessage::operator<<(const short& v);  // NOLINT
extern template LogMessage& LogMessage::operator<<(
    const unsigned short& v);  // NOLINT
extern template LogMessage& LogMessage::operator<<(const int& v);
extern template LogMessage& LogMessage::operator<<(const unsigned int& v);
extern template LogMessage& LogMessage::operator<<(const long& v);  // NOLINT
extern template LogMessage& LogMessage::operator<<(
    const unsigned long& v);  // NOLINT
extern template LogMessage& LogMessage::operator<<(
    const long long& v);  // NOLINT
extern template LogMessage& LogMessage::operator<<(
    const unsigned long long& v);  // NOLINT
extern template LogMessage& LogMessage::operator<<(void* const& v);
extern template LogMessage& LogMessage::operator<<(const void* const& v);
extern template LogMessage& LogMessage::operator<<(const float& v);
extern template LogMessage& LogMessage::operator<<(const double& v);
extern template LogMessage& LogMessage::operator<<(const bool& v);
extern template LogMessage& LogMessage::operator<<(const std::string& v);
extern template LogMessage& LogMessage::operator<<(const std::string_view& v);

// `LogMessageFatal` ensures the process will exit in failure after logging this
// message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) ABSL_ATTRIBUTE_COLD;
  LogMessageFatal(const char* file, int line,
                  std::string_view failure_msg) ABSL_ATTRIBUTE_COLD;
  ABSL_ATTRIBUTE_NORETURN ~LogMessageFatal();
};

class LogMessageQuietlyFatal : public LogMessage {
 public:
  LogMessageQuietlyFatal(const char* file, int line) ABSL_ATTRIBUTE_COLD;
  LogMessageQuietlyFatal(const char* file, int line,
                         std::string_view failure_msg) ABSL_ATTRIBUTE_COLD;
  ABSL_ATTRIBUTE_NORETURN ~LogMessageQuietlyFatal();
};

}  // namespace logging_internal
}  // namespace xls

#endif  // XLS_COMMON_LOGGING_LOG_MESSAGE_H_
