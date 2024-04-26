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

// Macros for non-fatal assertions.  The `XLS_RET_CHECK` family of macros return
// an absl::Status with code `absl::StatusCode::kInternal` from the current
// method.
//
//   XLS_RET_CHECK(ptr != nullptr);
//   XLS_RET_CHECK_GT(value, 0) << "Optional additional message";
//   XLS_RET_CHECK_FAIL() << "Always fails";
//   XLS_RET_CHECK_OK(status) << "If status is not OK, return an internal
//   error";
//
// The XLS_RET_CHECK* macros can only be used in functions that return
// absl::Status or absl::StatusOr.  The generated `absl::Status` will contain
// the string "XLS_RET_CHECK failure".
//
// On failure these routines will log a stack trace to `ERROR`.  The
// `XLS_RET_CHECK` macros end with a `StatusBuilder` in their tail position and
// can be customized like calls to `XLS_RETURN_IF_ERROR`.
//
// Be careful with the usage of XLS_RET_CHECK_* for checking user sensitive data
// since it logs the underlying input values on failure. XLS_RET_CHECK is a
// safer way for this situation because it just logs the input condition as a
// string literal on failure.

#ifndef XLS_COMMON_STATUS_RET_CHECK_H_
#define XLS_COMMON_STATUS_RET_CHECK_H_

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/source_location.h"
#include "xls/common/status/status_builder.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace internal_status_macros_ret_check {

// Returns a StatusBuilder that corresponds to a `XLS_RET_CHECK` failure.
xabsl::StatusBuilder RetCheckFailSlowPath(xabsl::SourceLocation location);
xabsl::StatusBuilder RetCheckFailSlowPath(xabsl::SourceLocation location,
                                          const char* condition);
xabsl::StatusBuilder RetCheckFailSlowPath(xabsl::SourceLocation location,
                                          const char* condition,
                                          const absl::Status& s);

// Takes ownership of `condition`.  This API is a little quirky because it is
// designed to make use of the `::Check_*Impl` methods that implement `CHECK_*`
// and `DCHECK_*`.
xabsl::StatusBuilder RetCheckFailSlowPath(xabsl::SourceLocation location,
                                          std::string* condition);

inline xabsl::StatusBuilder RetCheckImpl(const absl::Status& status,
                                         const char* condition,
                                         xabsl::SourceLocation location) {
  if (ABSL_PREDICT_TRUE(status.ok())) {
    return xabsl::StatusBuilder(absl::OkStatus(), location);
  }
  return RetCheckFailSlowPath(location, condition, status);
}

inline const absl::Status& AsStatus(const absl::Status& status) {
  return status;
}

template <typename T>
inline const absl::Status& AsStatus(const absl::StatusOr<T>& status_or) {
  return status_or.status();
}

// A helper class for formatting `expr (V1 vs. V2)` in a `XLS_RET_CHECK_XX`
// statement.  See `MakeCheckOpString` for sample usage.
class CheckOpMessageBuilder {
 public:
  // Inserts `exprtext` and ` (` to the stream.
  explicit CheckOpMessageBuilder(const char* exprtext);
  // Deletes `stream_`.
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream* ForVar1() { return stream_; }
  // For inserting the second variable (adds an intermediate ` vs. `).
  std::ostream* ForVar2();
  // Get the result (inserts the closing `)`).
  std::string* NewString();

 private:
  std::ostringstream* stream_;
};

// This formats a value for a failing `XLS_RET_CHECK_XX` statement.  Ordinarily,
// it uses the definition for `operator<<`, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
  (*os) << v;
}

// Overrides for char types provide readable values for unprintable characters.
void MakeCheckOpValueString(std::ostream* os, char v);
void MakeCheckOpValueString(std::ostream* os, signed char v);
void MakeCheckOpValueString(std::ostream* os, unsigned char v);

// We need an explicit specialization for `std::nullptr_t`.
void MakeCheckOpValueString(std::ostream* os, std::nullptr_t v);
void MakeCheckOpValueString(std::ostream* os, const char* v);
void MakeCheckOpValueString(std::ostream* os, const signed char* v);
void MakeCheckOpValueString(std::ostream* os, const unsigned char* v);
void MakeCheckOpValueString(std::ostream* os, char* v);
void MakeCheckOpValueString(std::ostream* os, signed char* v);
void MakeCheckOpValueString(std::ostream* os, unsigned char* v);

// Build the error message string.  Specify no inlining for code size.
template <typename T1, typename T2>
std::string* MakeCheckOpString(const T1& v1, const T2& v2,
                               const char* exprtext) ABSL_ATTRIBUTE_NOINLINE;

template <typename T1, typename T2>
std::string* MakeCheckOpString(const T1& v1, const T2& v2,
                               const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  ::xls::internal_status_macros_ret_check::MakeCheckOpValueString(
      comb.ForVar1(), v1);
  ::xls::internal_status_macros_ret_check::MakeCheckOpValueString(
      comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for `XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP`
// macro.  The `(int, int)` specialization works around the issue that the
// compiler will not instantiate the template version of the function on values
// of unnamed enum type - see comment below.
#define XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(name, op)                 \
  template <typename T1, typename T2>                                     \
  inline std::string* name##Impl(const T1& v1, const T2& v2,              \
                                 const char* exprtext) {                  \
    if (ABSL_PREDICT_TRUE(v1 op v2)) {                                    \
      return nullptr;                                                     \
    }                                                                     \
    return ::xls::internal_status_macros_ret_check::MakeCheckOpString(    \
        v1, v2, exprtext);                                                \
  }                                                                       \
  inline std::string* name##Impl(int v1, int v2, const char* exprtext) {  \
    return ::xls::internal_status_macros_ret_check::name##Impl<int, int>( \
        v1, v2, exprtext);                                                \
  }

XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(Check_EQ, ==)
XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(Check_NE, !=)
XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(Check_LE, <=)
XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(Check_LT, <)
XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(Check_GE, >=)
XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(Check_GT, >)
#undef XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP

// `XLS_RET_CHECK_EQ` and friends want to pass their arguments by reference,
// however this winds up exposing lots of cases where people have defined and
// initialized static const data members but never declared them (i.e. in a .cc
// file), meaning they are not referenceable.  This function avoids that problem
// for integers (the most common cases) by overloading for every primitive
// integer type, even the ones we discourage, and returning them by value.
template <typename T>
inline const T& GetReferenceableValue(const T& t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
inline signed char GetReferenceableValue(signed char t) { return t; }
inline short GetReferenceableValue(short t) {  // NOLINT: runtime/int
  return t;
}
inline unsigned short GetReferenceableValue(  // NOLINT: runtime/int
    unsigned short t) {                       // NOLINT: runtime/int
  return t;
}
inline int GetReferenceableValue(int t) { return t; }
inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
inline long GetReferenceableValue(long t)  // NOLINT: runtime/int
{
  return t;
}
inline unsigned long GetReferenceableValue(  // NOLINT: runtime/int
    unsigned long t) {                       // NOLINT: runtime/int
  return t;
}
inline long long GetReferenceableValue(long long t) {  // NOLINT: runtime/int
  return t;
}
inline unsigned long long GetReferenceableValue(  // NOLINT: runtime/int
    unsigned long long t) {                       // NOLINT: runtime/int
  return t;
}

}  // namespace internal_status_macros_ret_check
}  // namespace xls

#define XLS_RET_CHECK(cond)                                             \
  while (ABSL_PREDICT_FALSE(!(cond)))                                   \
  return ::xls::internal_status_macros_ret_check::RetCheckFailSlowPath( \
      XABSL_LOC, #cond)

#define XLS_RET_CHECK_FAIL()                                            \
  return ::xls::internal_status_macros_ret_check::RetCheckFailSlowPath( \
      XABSL_LOC)

// Takes an expression returning absl::Status and asserts that the status is
// `ok()`.  If not, it returns an internal error.
//
// This is similar to `XLS_RETURN_IF_ERROR` in that it propagates errors.  The
// difference is that it follows the behavior of `XLS_RET_CHECK`, returning an
// internal error (wrapping the original error text), including the filename and
// line number, and logging a stack trace.
//
// This is appropriate to use to write an assertion that a function that returns
// `absl::Status` cannot fail, particularly when the error code itself should
// not be surfaced.
#define XLS_RET_CHECK_OK(status)                                             \
  XLS_RETURN_IF_ERROR(::xls::internal_status_macros_ret_check::RetCheckImpl( \
      ::xls::internal_status_macros_ret_check::AsStatus(status), #status,    \
      XABSL_LOC))

#if defined(STATIC_ANALYSIS) || defined(PORTABLE_STATUS)
#define XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(name, op, lhs, rhs) \
  XLS_RET_CHECK((lhs)op(rhs))
#else
#define XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(name, op, lhs, rhs)           \
  while (                                                                     \
      std::string* _result =                                                  \
          ::xls::internal_status_macros_ret_check::Check_##name##Impl(        \
              ::xls::internal_status_macros_ret_check::GetReferenceableValue( \
                  lhs),                                                       \
              ::xls::internal_status_macros_ret_check::GetReferenceableValue( \
                  rhs),                                                       \
              #lhs " " #op " " #rhs))                                         \
  return ::xls::internal_status_macros_ret_check::RetCheckFailSlowPath(       \
      XABSL_LOC, _result)
#endif

#define XLS_RET_CHECK_EQ(lhs, rhs) \
  XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(EQ, ==, lhs, rhs)
#define XLS_RET_CHECK_NE(lhs, rhs) \
  XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(NE, !=, lhs, rhs)
#define XLS_RET_CHECK_LE(lhs, rhs) \
  XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(LE, <=, lhs, rhs)
#define XLS_RET_CHECK_LT(lhs, rhs) \
  XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(LT, <, lhs, rhs)
#define XLS_RET_CHECK_GE(lhs, rhs) \
  XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(GE, >=, lhs, rhs)
#define XLS_RET_CHECK_GT(lhs, rhs) \
  XLS_COMMON_MACROS_INTERNAL_RET_CHECK_OP(GT, >, lhs, rhs)

#endif  // XLS_COMMON_STATUS_RET_CHECK_H_
