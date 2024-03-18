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

// NOTE: Because Abseil has released logging support, these macros mostly (with
// the notable exception of VLOG() and friends) directly wrap Abseil macros.
// They still exist because historically we needed our own logging
// implementation before Abseil released logging. XLS developers should continue
// to use the XLS_-prefixed macros for now.

// TODO: google/xls#1318 - remove these aliases.
//
// Defines a set of logging macros and related APIs.  The two most basic
// invocations look like this:
//
//   XLS_LOG(INFO) << "Found " << num_cookies << " cookies";
//
//   // This one is an assertion; if there's no more cheese the program logs
//   // this message and terminates with a non-zero exit code.
//   CHECK(!cheese.empty()) << "Out of Cheese";
//
// Except where noted, no interfaces in this header are async-signal-safe; their
// use in signal handlers is unsupported and may deadlock your program or eat
// your lunch.
//
// Many logging statements are inherently conditional.  For example,
// `CHECK(foo)` and `LOG_IF(INFO, !foo)` do nothing if `foo` is true.
// Even seemingly unconditional statements like `XLS_LOG(INFO)` might be
// disabled at compile-time to minimize binary size or for security reasons.
//
// * Except for the condition in a `CHECK` or `QCHECK` statement,
//   programs must not rely on evaluation of expressions anywhere in logging
//   statements for correctness.  For example, this is ok:
//
//     CHECK((fp = fopen("config.ini", "r")) != nullptr);
//
//   But this is probably not ok:
//
//     XLS_LOG(INFO) << "Server status: " << StartServerAndReturnStatusString();
//
//   This is bad too; the `i++` in the `LOG_IF` condition may not be
//   evaluated, which would make the loop infinite:
//
//     for (int i = 0; i < 1000000;)
//       LOG_IF(INFO, i++ % 1000 == 0) << "Still working...";
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
//       LOG_IF(ERROR, false) << "The fox says " << fox_line1;
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

// IWYU pragma: begin_exports
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/common/logging/log_flags.h"
// IWYU pragma: end_exports

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
//   * .AtLocation(std::string_view file, int line)
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
#define XLS_LOG(severity) LOG(severity)

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
#define XLS_VLOG(verbose_level) VLOG(verbose_level)

#endif  // XLS_COMMON_LOGGING_LOGGING_H_
