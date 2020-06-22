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

// Defines a set of logging macros and related APIs.  The two most basic
// invocations look like this:
//
//   XLS_LOG(INFO) << "Found " << num_cookies << " cookies";
//
//   // This one is an assertion; if there's no more cheese the program logs
//   // this message and terminates with a non-zero exit code.
//   XLS_CHECK(!cheese.empty()) << "Out of Cheese";
//
// Except where noted, no interfaces in this header are async-signal-safe; their
// use in signal handlers is unsupported and may deadlock your program or eat
// your lunch.
//
// Many logging statements are inherently conditional.  For example,
// `XLS_CHECK(foo)` and `XLS_LOG_IF(INFO, !foo)` do nothing if `foo` is true.
// Even seemingly unconditional statements like `XLS_LOG(INFO)` might be
// disabled at compile-time to minimize binary size or for security reasons.
//
// * Except for the condition in a `XLS_CHECK` or `XLS_QCHECK` statement,
//   programs must not rely on evaluation of expressions anywhere in logging
//   statements for correctness.  For example, this is ok:
//
//     XLS_CHECK((fp = fopen("config.ini", "r")) != nullptr);
//
//   But this is probably not ok:
//
//     XLS_LOG(INFO) << "Server status: " << StartServerAndReturnStatusString();
//
//   This is bad too; the `i++` in the `XLS_LOG_IF` condition may not be
//   evaluated, which would make the loop infinite:
//
//     for (int i = 0; i < 1000000;)
//       XLS_LOG_IF(INFO, i++ % 1000 == 0) << "Still working...";
//
// * Except where otherwise noted, conditions which cause a statement not to log
//   also cause expressions not to be evaluated.  Programs may rely on this for
//   performance reasons, e.g. by streaming the result of an expensive function
//   call into an `XLS_LOG` statement.
// * Care has been taken to ensure that expressions are parsed by the compiler
//   even if they are never evaluated.  This means that syntax errors will be
//   caught and variables will be considered used for the purposes of
//   unused-variable diagnostics.  For example, this statement won't compile
//   even if `INFO`-level logging has been compiled out:
//
//     int number_of_cakes = 40;
//     XLS_LOG(INFO) << "Number of cakes: " << num_of_cakes;  // Note the typo!
//
//   Similarly, this won't produce unused-variable compiler diagnostics even
//   if `INFO`-level logging is compiled out:
//
//     {
//       char fox_line1[] = "Hatee-hatee-hatee-ho!";
//       XLS_LOG_IF(ERROR, false) << "The fox says " << fox_line1;
//       char fox_line2[] = "A-oo-oo-oo-ooo!";
//       XLS_LOG(INFO) << "The fox also says " << fox_line2;
//     }
//
//   This error-checking is not perfect; for example, symbols that have been
//   declared but not defined may not produce link errors if used in logging
//   statements that compile away.
//
// Expressions streamed into these macros are formatted using `operator<<` just
// as they would be if streamed into a `std::ostream`, however it should be
// noted that their actual type is unspecified.
//
// To implement a custom formatting operator for a type you own, define
// `std::ostream& operator<<(std::ostream&, ...)` in your type's namespace (for
// ADL) just as you would to stream it into `std::cout`.
//
// Those macros that support streaming honor output manipulators and `fmtflag`
// changes that output data (e.g. `std::ends`) or control formatting of data
// (e.g. `std::hex` and `std::fixed`), however flushing such a stream is
// ignored.  The message produced by a log statement is sent to registered
// `LogSink` instances at the end of the statement; those sinks are responsible
// for their own flushing (e.g. to disk) semantics.
//
// Flag settings are not carried over from one `XLS_LOG` statement to the next;
// this is a bit different than e.g. `std::cout`:
//
//   XLS_LOG(INFO) << std::hex << 0xdeadbeef;  // logs "0xdeadbeef"
//   XLS_LOG(INFO) << 0xdeadbeef;              // logs "3735928559"

#ifndef XLS_COMMON_LOGGING_LOGGING_H_
#define XLS_COMMON_LOGGING_LOGGING_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "xls/common/logging/check_ops.h"
#include "xls/common/logging/condition.h"
#include "xls/common/logging/log_sink.h"
#include "xls/common/logging/logging_internal.h"
#include "xls/common/logging/vlog_is_on.h"

// -----------------------------------------------------------------------------
// `XLS_LOG` Macros
// -----------------------------------------------------------------------------
//
// Most `XLS_LOG` macros take a severity level argument.  The severity levels
// are `INFO`, `WARNING`, `ERROR`, and `FATAL`.  They are defined in
// log_severity.h.
// * The `FATAL` severity level terminates the program with a stack trace after
//   logging its message.
// * The `QFATAL` pseudo-severity level is equivalent to `FATAL` but triggers
//   quieter termination messages, i.e. without a full stack trace, and skips
//   running registered error handlers.
//
// Some preprocessor shenanigans are used to ensure that e.g. `XLS_LOG(INFO)`
// has the same meaning even if a local symbol or preprocessor macro named
// `INFO` is defined.  To specify a severity level using an expression instead
// of a literal, use `LEVEL(expr)`. Example:
//
//   XLS_LOG(
//       LEVEL(stale ? absl::LogSeverity::kWarning : absl::LogSeverity::kInfo))
//       << "Cookies are " << days << " days old";

// `XLS_LOG` macros evaluate to an unterminated statement.  The value at the end
// of the statement supports some chainable methods:
//
//   * .AtLocation(absl::string_view file, int line)
//     .AtLocation(xabsl::SourceLocation loc)
//     Overrides the location inferred from the callsite.  The string pointed to
//     by `file` must be valid until the end of the statement.
//   * .NoPrefix()
//     Omits the prefix from this line.  The prefix includes metadata about the
//     logged data such as source code location and timestamp.
//   * .WithPerror()
//     Appends to the logged message a colon, a space, a textual description of
//     the current value of `errno` (as by `strerror(3)`), and the numerical
//     value of `errno`.
//   * .WithVerbosity(int verbose_level)
//     Sets the verbosity field of the logged message as if it was logged by
//     `XLS_VLOG(verbose_level)`.  Unlike `XLS_VLOG`, this method does not
//     affect evaluation of the statement when the specified `verbose_level` has
//     been disabled.  The only effect is on `LogSink` implementations which
//     make use of the `absl::LogSink::verbosity()` value.  The value
//     `absl::LogEntry::kNoVerboseLevel` can be specified to mark the message
//     not verbose.
//   * .ToSinkAlso(absl::LogSink* sink)
//     Sends this message to `*sink` in addition to whatever other sinks it
//     would otherwise have been sent to.  `sink` must not be null.
//   * .ToSinkOnly(absl::LogSink* sink)
//     Sends this message to `*sink` and no others.  `sink` must not be null.

// `XLS_LOG` takes a single argument which is a severity level.  Data streamed
// in comprise the logged message. Example:
//
//   XLS_LOG(INFO) << "Found " << num_cookies << " cookies";
#define XLS_LOG(severity) \
  switch (0)              \
  default:                \
    XLS_LOGGING_INTERNAL_LOG_##severity.stream()

// `XLS_VLOG` uses numeric levels to provide verbose logging that can configured
// at runtime, including at a per-module level.  `XLS_VLOG` statements are
// logged at `INFO` severity if they are logged at all; the numeric levels are
// on a different scale than the proper severity levels.  Positive levels are
// disabled by default.  Negative levels should not be used.
// Example:
//
//   XLS_VLOG(1) << "I print when you run the program with --v=1 or higher";
//   XLS_VLOG(2) << "I print when you run the program with --v=2 or higher";
//
// See vlog_is_on.h for further documentation, including the usage of the
// --vmodule flag to log at different levels in different source files.
#define XLS_VLOG(verbose_level) XLS_VLOG_IF(verbose_level, true)

// `XLS_LOG_IF` and friends add a second argument which specifies a condition.
// If the condition is false, nothing is logged. Example:
//
//   XLS_LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";
#define XLS_LOG_IF(severity, condition)       \
  switch (0)                                  \
  default:                                    \
    XLS_LOGGING_INTERNAL_CONDITION(condition) \
  XLS_LOGGING_INTERNAL_LOG_##severity.stream()
#define XLS_VLOG_IF(verbose_level, condition)                     \
  switch (0)                                                      \
  default:                                                        \
    XLS_LOGGING_INTERNAL_CONDITION((condition) &&                 \
                                   XLS_VLOG_IS_ON(verbose_level)) \
  XLS_LOGGING_INTERNAL_LOG_INFO.WithVerbosity(verbose_level).stream()

// -----------------------------------------------------------------------------
// `XLS_CHECK` Macros
// -----------------------------------------------------------------------------
//
// `XLS_CHECK` macros terminate the program with a fatal error if the specified
// condition is not true.  Except for `XLS_DCHECK`, they are not controlled by
// `NDEBUG` (cf. `assert`), so the check will be executed regardless of
// compilation mode.  `XLS_CHECK` and friends are thus useful for confirming
// invariants in situations where continuing to run would be worse than
// crashing, e.g. due to risk of data corruption or security compromise.  It can
// also be useful to deliberately terminate at a particular place with a useful
// message and backtrace instead of relying on an expected segmentation fault.

// `XLS_CHECK` terminates the program with a fatal error if condition is not
// true. Example:
//
//   XLS_CHECK(!cheese.empty()) << "Out of Cheese";
//
// Might produce a message like:
//
//   Check failed: !cheese.empty() Out of Cheese
#define XLS_CHECK(condition)                                         \
  switch (0)                                                         \
  default:                                                           \
    XLS_LOGGING_INTERNAL_CONDITION(ABSL_PREDICT_FALSE(!(condition))) \
  XLS_LOGGING_INTERNAL_CHECK(#condition).stream()

// `XLS_QCHECK` behaves like `XLS_CHECK` but does not print a full stack trace
// and does not run registered error handlers (as `QFATAL`).  It is useful when
// the problem is definitely unrelated to program flow, e.g. when validating
// user input.
#define XLS_QCHECK(condition)                                        \
  switch (0)                                                         \
  default:                                                           \
    XLS_LOGGING_INTERNAL_CONDITION(ABSL_PREDICT_FALSE(!(condition))) \
  XLS_LOGGING_INTERNAL_QCHECK(#condition).stream()

// `XLS_DCHECK` behaves like `XLS_CHECK` in debug mode and does nothing
// otherwise.  Unlike with `XLS_CHECK`, it is not safe to rely on evaluation of
// `condition`.
#ifndef NDEBUG
#define XLS_DCHECK(condition) XLS_CHECK(condition)
#else
#define XLS_DCHECK(condition)  \
  while (false && (condition)) \
  ::xls::logging_internal::NullStreamFatal().stream()
#endif

// `XLS_CHECK_EQ` and friends are syntactic sugar for `XLS_CHECK(x == y)` that
// automatically output the expression being tested and the evaluated values on
// either side. Example:
//
//   int x = 3, y = 5;
//   XLS_CHECK_EQ(2 * x, y) << "oops!";
//
// Might produce a message like:
//
//   Check failed: 2 * x == y (6 vs. 5) oops!
//
// The values must implement the appropriate comparison operator as well as
// `operator<<(std::ostream&, ...)`.  Care is taken to ensure that each
// argument is evaluated exactly once, and that anything which is legal to pass
// as a function argument is legal here.  In particular, the arguments may be
// temporary expressions which will end up being destroyed at the end of the
// statement. Example:
//
//   XLS_CHECK_EQ(std::string("abc")[1], 'b');
//
// WARNING: Passing `NULL` as an argument to `XLS_CHECK_EQ` and similar macros
// does not compile.  Use `nullptr` instead.
#define XLS_CHECK_EQ(val1, val2) \
  XLS_LOGGING_INTERNAL_CHECK_OP(Check_EQ, ==, val1, val2)
#define XLS_CHECK_NE(val1, val2) \
  XLS_LOGGING_INTERNAL_CHECK_OP(Check_NE, !=, val1, val2)
#define XLS_CHECK_LE(val1, val2) \
  XLS_LOGGING_INTERNAL_CHECK_OP(Check_LE, <=, val1, val2)
#define XLS_CHECK_LT(val1, val2) \
  XLS_LOGGING_INTERNAL_CHECK_OP(Check_LT, <, val1, val2)
#define XLS_CHECK_GE(val1, val2) \
  XLS_LOGGING_INTERNAL_CHECK_OP(Check_GE, >=, val1, val2)
#define XLS_CHECK_GT(val1, val2) \
  XLS_LOGGING_INTERNAL_CHECK_OP(Check_GT, >, val1, val2)
#define XLS_QCHECK_EQ(val1, val2) \
  XLS_LOGGING_INTERNAL_QCHECK_OP(Check_EQ, ==, val1, val2)
#define XLS_QCHECK_NE(val1, val2) \
  XLS_LOGGING_INTERNAL_QCHECK_OP(Check_NE, !=, val1, val2)
#define XLS_QCHECK_LE(val1, val2) \
  XLS_LOGGING_INTERNAL_QCHECK_OP(Check_LE, <=, val1, val2)
#define XLS_QCHECK_LT(val1, val2) \
  XLS_LOGGING_INTERNAL_QCHECK_OP(Check_LT, <, val1, val2)
#define XLS_QCHECK_GE(val1, val2) \
  XLS_LOGGING_INTERNAL_QCHECK_OP(Check_GE, >=, val1, val2)
#define XLS_QCHECK_GT(val1, val2) \
  XLS_LOGGING_INTERNAL_QCHECK_OP(Check_GT, >, val1, val2)

#ifndef NDEBUG
#define XLS_DCHECK_EQ(val1, val2) XLS_CHECK_EQ(val1, val2)
#define XLS_DCHECK_NE(val1, val2) XLS_CHECK_NE(val1, val2)
#define XLS_DCHECK_LE(val1, val2) XLS_CHECK_LE(val1, val2)
#define XLS_DCHECK_LT(val1, val2) XLS_CHECK_LT(val1, val2)
#define XLS_DCHECK_GE(val1, val2) XLS_CHECK_GE(val1, val2)
#define XLS_DCHECK_GT(val1, val2) XLS_CHECK_GT(val1, val2)
#else
#define XLS_DCHECK_EQ(val1, val2) XLS_LOGGING_INTERNAL_DCHECK_NOP(val1, val2)
#define XLS_DCHECK_NE(val1, val2) XLS_LOGGING_INTERNAL_DCHECK_NOP(val1, val2)
#define XLS_DCHECK_LE(val1, val2) XLS_LOGGING_INTERNAL_DCHECK_NOP(val1, val2)
#define XLS_DCHECK_LT(val1, val2) XLS_LOGGING_INTERNAL_DCHECK_NOP(val1, val2)
#define XLS_DCHECK_GE(val1, val2) XLS_LOGGING_INTERNAL_DCHECK_NOP(val1, val2)
#define XLS_DCHECK_GT(val1, val2) XLS_LOGGING_INTERNAL_DCHECK_NOP(val1, val2)
#endif

// absl::Status success comparison.
// This is better than CHECK((val).ok()) because the embedded
// error string gets printed by the CHECK_EQ.
#define XLS_CHECK_OK(val) XLS_CHECK_EQ(::absl::OkStatus(), (val))
#define XLS_QCHECK_OK(val) XLS_QCHECK_EQ(::absl::OkStatus(), (val))
#define XLS_DCHECK_OK(val) XLS_DCHECK_EQ(::absl::OkStatus(), (val))

// `XLS_DIE_IF_NULL` behaves as `XLS_CHECK_NE` vs `nullptr` but *also* "returns"
// its argument.  It is useful in initializers where statements (like
// `XLS_CHECK_NE`) can't be used.  Outside initializers, prefer `XLS_CHECK` or
// `XLS_CHECK_NE`. `XLS_DIE_IF_NULL` works for both raw pointers and
// (compatible) smart pointers including `std::unique_ptr` and
// `std::shared_ptr`; more generally, it works for any type that can be compared
// to nullptr_t.  For types that aren't raw pointers, `XLS_DIE_IF_NULL` returns
// a reference to its argument, preserving the value category. Example:
//
//   Foo() : bar_(XLS_DIE_IF_NULL(MethodReturningUniquePtr())) {}
//
// Use `XLS_CHECK(ptr != nullptr)` if the returned pointer is
// unused.
#define XLS_DIE_IF_NULL(val) \
  ::xls::logging_internal::DieIfNull(__FILE__, __LINE__, #val, (val))

namespace xls {

// Add or remove a `LogSink` as a consumer of logging data.  Thread-safe.
void AddLogSink(LogSink* sink);
void RemoveLogSink(LogSink* sink);

}  // namespace xls

#endif  // XLS_COMMON_LOGGING_LOGGING_H_
