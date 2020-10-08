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

#ifndef XLS_COMMON_LOGGING_CHECK_OPS_H_
#define XLS_COMMON_LOGGING_CHECK_OPS_H_

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "xls/common/logging/null_guard.h"

#ifdef NDEBUG
// `NDEBUG` is defined, so `XLS_DCHECK_EQ(x, y)` and so on do nothing.  However,
// we still want the compiler to parse `x` and `y`, because we don't want to
// lose potentially useful errors and warnings.
#define XLS_LOGGING_INTERNAL_DCHECK_NOP(x, y) \
  while (false && ((void)(x), (void)(y), 0))  \
  ::xls::logging_internal::NullStream().stream()
#endif

#define XLS_LOGGING_INTERNAL_CHECK(failure_message) \
  ::xls::logging_internal::LogMessageFatal(__FILE__, __LINE__, failure_message)
#define XLS_LOGGING_INTERNAL_QCHECK(failure_message)                  \
  ::xls::logging_internal::LogMessageQuietlyFatal(__FILE__, __LINE__, \
                                                  failure_message)

#define XLS_LOGGING_INTERNAL_CHECK_OP(name, op, val1, val2)                   \
  while (std::string* xls_logging_internal_check_op_result                    \
             ABSL_ATTRIBUTE_UNUSED = ::xls::logging_internal::name##Impl(     \
                 ::xls::logging_internal::GetReferenceableValue(val1),        \
                 ::xls::logging_internal::GetReferenceableValue(val2),        \
                 XLS_LOGGING_INTERNAL_STRIP_STRING_LITERAL(#val1 " " #op      \
                                                                 " " #val2))) \
  XLS_LOGGING_INTERNAL_CHECK(*xls_logging_internal_check_op_result).stream()
#define XLS_LOGGING_INTERNAL_QCHECK_OP(name, op, val1, val2)                  \
  while (std::string* xls_logging_internal_qcheck_op_result =                 \
             ::xls::logging_internal::name##Impl(                             \
                 ::xls::logging_internal::GetReferenceableValue(val1),        \
                 ::xls::logging_internal::GetReferenceableValue(val2),        \
                 XLS_LOGGING_INTERNAL_STRIP_STRING_LITERAL(#val1 " " #op      \
                                                                 " " #val2))) \
  XLS_LOGGING_INTERNAL_QCHECK(*xls_logging_internal_qcheck_op_result).stream()

namespace xls {
namespace logging_internal {
namespace detect_specialization {

// MakeCheckOpString is being specialized for every T and U pair that is being
// passed to the CHECK_op macros. However, there is a lot of redundancy in these
// specializations that creates unnecessary library and binary bloat.
// The number of instantiations tends to be O(n^2) because we have two
// independent inputs. This technique works by reducing `n`.
//
// Most user-defined types being passed to CHECK_op end up being printed as a
// builtin type. For example, enums tend to be implicitly converted to its
// underlying type when calling operator<<, and pointers are printed with the
// `const void*` overload.
// To reduce the number of instantiations we coerce these values before calling
// MakeCheckOpString instead of inside it.
//
// To detect if this coercion is needed, we duplicate all the relevant
// operator<< overloads as specified in the standard, just in a different
// namespace. If the call to `stream << value` becomes ambiguous, it means that
// one of these overloads is the one selected by overload resolution. We then
// do overload resolution again just with our overload set to see which one gets
// selected. That tells us which type to coerce to.
// If the augmented call was not ambiguous, it means that none of these were
// selected and we can't coerce the input.
//
// As a secondary step to reduce code duplication, we promote integral types to
// their 64-bit variant. This does not change the printed value, but reduces the
// number of instantiations even further. Promoting an integer is very cheap at
// the call site.
int64_t operator<<(std::ostream&, short value);           // NOLINT
int64_t operator<<(std::ostream&, unsigned short value);  // NOLINT
int64_t operator<<(std::ostream&, int value);
int64_t operator<<(std::ostream&, unsigned int value);
int64_t operator<<(std::ostream&, long value);                 // NOLINT
uint64_t operator<<(std::ostream&, unsigned long value);       // NOLINT
int64_t operator<<(std::ostream&, long long value);            // NOLINT
uint64_t operator<<(std::ostream&, unsigned long long value);  // NOLINT
float operator<<(std::ostream&, float value);
double operator<<(std::ostream&, double value);
long double operator<<(std::ostream&, long double value);
bool operator<<(std::ostream&, bool value);
const void* operator<<(std::ostream&, const void* value);
const void* operator<<(std::ostream&, std::nullptr_t);

// These `char` overloads are specified like this in the standard, so we have to
// write them exactly the same to ensure the call is ambiguous.
// If we wrote it in a different way (eg taking std::ostream instead of the
// template) then one call might have a higher rank than the other and it would
// not be ambiguous.
template <typename Traits>
char operator<<(std::basic_ostream<char, Traits>&, char);
template <typename Traits>
signed char operator<<(std::basic_ostream<char, Traits>&, signed char);
template <typename Traits>
unsigned char operator<<(std::basic_ostream<char, Traits>&, unsigned char);
template <typename Traits>
const char* operator<<(std::basic_ostream<char, Traits>&, const char*);
template <typename Traits>
const signed char* operator<<(std::basic_ostream<char, Traits>&,
                              const signed char*);
template <typename Traits>
const unsigned char* operator<<(std::basic_ostream<char, Traits>&,
                                const unsigned char*);

// This overload triggers when the call is not ambiguous.
// It means that T is being printed with some overload not on this list.
// We keep the value as `const T&`.
template <typename T, typename = decltype(std::declval<std::ostream&>()
                                          << std::declval<const T&>())>
const T& Detect(int);

// This overload triggers when the call is ambiguous.
// It means that T is either one from this list or printed as one from this
// list. Eg an enum that decays to `int` for printing.
// We ask the overload set to give us the type we want to convert it to.
template <typename T>
decltype(detect_specialization::operator<<(std::declval<std::ostream&>(),
                                           std::declval<const T&>()))
Detect(char);

}  // namespace detect_specialization

template <typename T>
using CheckOpStreamType = decltype(detect_specialization::Detect<T>(0));

// A helper class for formatting `expr (V1 vs. V2)` in a `CHECK_XX` statement.
// See `MakeCheckOpString` for sample usage.
class CheckOpMessageBuilder {
 public:
  // Inserts `exprtext` and ` (` to the stream.
  explicit CheckOpMessageBuilder(const char* exprtext);
  // For inserting the first variable.
  std::ostream* ForVar1() { return stream_.get(); }
  // For inserting the second variable (adds an intermediate ` vs. `).
  std::ostream* ForVar2();
  // Get the result (inserts the closing `)`).
  std::string* NewString();

 private:
  std::unique_ptr<std::ostringstream> stream_;
};

// This formats a value for a failing `CHECK_XX` statement.  Ordinarily, it uses
// the definition for `operator<<`, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
  *os << NullGuard<T>::Guard(v);
}

// Build the error message string.  Specify no inlining for code size.
template <typename T1, typename T2>
std::string* MakeCheckOpString(T1 v1, T2 v2,
                               const char* exprtext) ABSL_ATTRIBUTE_NOINLINE;

template <typename T1, typename T2>
std::string* MakeCheckOpString(T1 v1, T2 v2, const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for `XLS_LOGGING_INTERNAL_CHECK_OP` macro family.  The
// `(int, int)` override works around the issue that the compiler will not
// instantiate the template version of the function on values of unnamed enum
// type.
#define XLS_LOGGING_INTERNAL_CHECK_OP_IMPL(name, op)                     \
  template <typename T1, typename T2>                                    \
  inline std::string* name##Impl(const T1& v1, const T2& v2,             \
                                 const char* exprtext) {                 \
    if (ABSL_PREDICT_TRUE(v1 op v2)) return nullptr;                     \
    using U1 = CheckOpStreamType<T1>;                                    \
    using U2 = CheckOpStreamType<T2>;                                    \
    return MakeCheckOpString<U1, U2>(v1, v2, exprtext);                  \
  }                                                                      \
  inline std::string* name##Impl(int v1, int v2, const char* exprtext) { \
    return name##Impl<int, int>(v1, v2, exprtext);                       \
  }

XLS_LOGGING_INTERNAL_CHECK_OP_IMPL(Check_EQ, ==)
XLS_LOGGING_INTERNAL_CHECK_OP_IMPL(Check_NE, !=)
XLS_LOGGING_INTERNAL_CHECK_OP_IMPL(Check_LE, <=)
XLS_LOGGING_INTERNAL_CHECK_OP_IMPL(Check_LT, <)
XLS_LOGGING_INTERNAL_CHECK_OP_IMPL(Check_GE, >=)
XLS_LOGGING_INTERNAL_CHECK_OP_IMPL(Check_GT, >)
#undef XLS_LOGGING_INTERNAL_CHECK_OP_IMPL

// `XLS_CHECK_EQ` and friends want to pass their arguments by reference, however
// this winds up exposing lots of cases where people have defined and
// initialized static const data members but never declared them (i.e. in a .cc
// file), meaning they are not referenceable.  This function avoids that problem
// for integers (the most common cases) by overloading for every primitive
// integer type, and returning them by value.
template <typename T>
inline const T& GetReferenceableValue(const T& t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
inline signed char GetReferenceableValue(signed char t) { return t; }
inline short GetReferenceableValue(short t) { return t; }        // NOLINT
inline unsigned short GetReferenceableValue(unsigned short t) {  // NOLINT
  return t;
}
inline int GetReferenceableValue(int t) { return t; }
inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
inline long GetReferenceableValue(long t) { return t; }        // NOLINT
inline unsigned long GetReferenceableValue(unsigned long t) {  // NOLINT
  return t;
}
inline long long GetReferenceableValue(long long t) { return t; }  // NOLINT
inline unsigned long long GetReferenceableValue(                   // NOLINT
    unsigned long long t) {                                        // NOLINT
  return t;
}

}  // namespace logging_internal
}  // namespace xls

#endif  // XLS_COMMON_LOGGING_CHECK_OPS_H_
