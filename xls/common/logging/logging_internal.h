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

#ifndef XLS_COMMON_LOGGING_LOGGING_INTERNAL_H_
#define XLS_COMMON_LOGGING_LOGGING_INTERNAL_H_

#include "absl/base/log_severity.h"
#include "xls/common/logging/log_message.h"

// `XLS_LOGGING_INTERNAL_STRIP_STRING_LITERAL` wraps string literals that
// should be stripped in `STRIP_LOG` mode.
#if STRIP_LOG
#define XLS_LOGGING_INTERNAL_STRIP_STRING_LITERAL(literal) ""
#else
#define XLS_LOGGING_INTERNAL_STRIP_STRING_LITERAL(literal) literal
#endif

#define XLS_LOGGING_INTERNAL_SEVERITY_INFO ::absl::LogSeverity::kInfo
#define XLS_LOGGING_INTERNAL_SEVERITY_WARNING ::absl::LogSeverity::kWarning
#define XLS_LOGGING_INTERNAL_SEVERITY_ERROR ::absl::LogSeverity::kError
#define XLS_LOGGING_INTERNAL_SEVERITY_FATAL ::absl::LogSeverity::kFatal
#define XLS_LOGGING_INTERNAL_SEVERITY_LEVEL(severity) (severity)

#define XLS_LOGGING_INTERNAL_LOG_INFO                     \
  ::xls::logging_internal::LogMessage(__FILE__, __LINE__, \
                                      ::absl::LogSeverity::kInfo)
#define XLS_LOGGING_INTERNAL_LOG_WARNING                  \
  ::xls::logging_internal::LogMessage(__FILE__, __LINE__, \
                                      ::absl::LogSeverity::kWarning)
#define XLS_LOGGING_INTERNAL_LOG_ERROR                    \
  ::xls::logging_internal::LogMessage(__FILE__, __LINE__, \
                                      ::absl::LogSeverity::kError)
#define XLS_LOGGING_INTERNAL_LOG_FATAL \
  ::xls::logging_internal::LogMessageFatal(__FILE__, __LINE__)
#define XLS_LOGGING_INTERNAL_LOG_QFATAL \
  ::xls::logging_internal::LogMessageQuietlyFatal(__FILE__, __LINE__)

#ifdef NDEBUG
#define XLS_LOGGING_INTERNAL_LOG_DFATAL                   \
  ::xls::logging_internal::LogMessage(__FILE__, __LINE__, \
                                      ::absl::LogSeverity::kError)
#else
#define XLS_LOGGING_INTERNAL_LOG_DFATAL \
  ::xls::logging_internal::LogMessageFatal(__FILE__, __LINE__)
#endif
#define XLS_LOGGING_INTERNAL_LOG_LEVEL(severity)          \
  ::xls::logging_internal::LogMessage(__FILE__, __LINE__, \
                                      ::absl::NormalizeLogSeverity(severity))

namespace xls {
namespace logging_internal {

// Crashes the process after logging `expertext` annotated at the `file` and
// `line` location. Called when `XLS_DIE_IF_NULL` fails. Calling this function
// generates less code than its implementation would if inlined, for a slight
// code size reduction each time `XLS_DIE_IF_NULL` is called.
ABSL_ATTRIBUTE_NORETURN ABSL_ATTRIBUTE_NOINLINE void DieBecauseNull(
    const char* file, int line, const char* exprtext);

// Helper for `XLS_DIE_IF_NULL`.
//
// In C++11, all cases can be handled by a single function. Since the value
// category of the argument is preserved (also for rvalue references), member
// initializer lists like the one below will compile correctly:
//
//   Foo() : bar_(XLS_DIE_IF_NULL(MethodReturningUniquePtr())) {}
template <typename T>
ABSL_MUST_USE_RESULT T DieIfNull(const char* file, int line,
                                 const char* exprtext, T&& t) {
  if (ABSL_PREDICT_FALSE(t == nullptr)) {
    // Call a non-inline helper function for a small code size improvement.
    DieBecauseNull(file, line, exprtext);
  }
  return std::forward<T>(t);
}

}  // namespace logging_internal
}  // namespace xls

#endif  // XLS_COMMON_LOGGING_LOGGING_INTERNAL_H_
