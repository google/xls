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
//
// API for capturing source-code location information.
// Based on http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4519.pdf.
//
// To define a function that has access to the source location of the
// callsite, define it with a parameter of type `xabsl::SourceLocation`. The
// caller can then invoke the function, passing `XABSL_LOC` as the argument.
//
// If at all possible, make the `xabsl::SourceLocation` parameter be the
// function's last parameter. That way, when `std::source_location` is
// available, you will be able to switch to it, and give the parameter a default
// argument of `std::source_location::current()`. Users will then be able to
// omit that argument, and the default will automatically capture the location
// of the callsite.

#ifndef XLS_COMMON_SOURCE_LOCATION_H_
#define XLS_COMMON_SOURCE_LOCATION_H_

#ifdef XLS_USE_ABSL_SOURCE_LOCATION
#include "absl/types/source_location.h"

// Use absl::SourceLocation
namespace xabsl {
using SourceLocation = absl::SourceLocation;
#define XABSL_LOC ABSL_LOC
#define XABSL_LOC_CURRENT_DEFAULT_ARG ABSL_LOC_CURRENT_DEFAULT_ARG
}  // namespace xabsl

#else
#include <cstdint>

// OSS Version of source_location

#define XABSL_HAVE_SOURCE_LOCATION_CURRENT 1

// The xabsl namespace has types that are anticipated to become available in
// Abseil reasonably soon, at which point they can be removed. These types are
// not in the xls namespace to make it easier to search/replace migrate usages
// to Abseil in the future.
namespace xabsl {

// Class representing a specific location in the source code of a program.
// `xabsl::SourceLocation` is copyable.
class SourceLocation {
  struct PrivateTag {
   private:
    explicit PrivateTag() = default;
    friend class SourceLocation;
  };

 public:
  // Avoid this constructor; it populates the object with dummy values.
  constexpr SourceLocation() : line_(0), file_name_(nullptr) {}

  // Wrapper to invoke the private constructor below. This should only be used
  // by the `XABSL_LOC` macro, hence the name.
  static constexpr SourceLocation DoNotInvokeDirectly(std::uint_least32_t line,
                                                      const char* file_name) {
    return SourceLocation(line, file_name);
  }

#ifdef XABSL_HAVE_SOURCE_LOCATION_CURRENT
  // SourceLocation::current
  //
  // Creates a `SourceLocation` based on the current line and file.  APIs that
  // accept a `SourceLocation` as a default parameter can use this to capture
  // their caller's locations.
  //
  // Example:
  //
  //   void TracedAdd(int i, SourceLocation loc = SourceLocation::current()) {
  //     std::cout << loc.file_name() << ":" << loc.line() << " added " << i;
  //     ...
  //   }
  //
  //   void UserCode() {
  //     TracedAdd(1);
  //     TracedAdd(2);
  //   }
  static constexpr SourceLocation current(
      PrivateTag = PrivateTag{}, std::uint_least32_t line = __builtin_LINE(),
      const char* file_name = __builtin_FILE()) {
    return SourceLocation(line, file_name);
  }
#else
  // Creates a dummy `SourceLocation` of "<source_location>" at line number 1,
  // if no `SourceLocation::current()` implementation is available.
  static constexpr SourceLocation current() {
    return SourceLocation(1, "<source_location>");
  }
#endif
  // The line number of the captured source location.
  constexpr std::uint_least32_t line() const { return line_; }

  // The file name of the captured source location.
  constexpr const char* file_name() const { return file_name_; }

  // `column()` and `function_name()` are omitted because we don't have a way to
  // support them.

 private:
  // Do not invoke this constructor directly. Instead, use the `XABSL_LOC` macro
  // below.
  //
  // `file_name` must outlive all copies of the `xabsl::SourceLocation` object,
  // so in practice it should be a string literal.
  constexpr SourceLocation(std::uint_least32_t line, const char* file_name)
      : line_(line), file_name_(file_name) {}

  friend constexpr int UseUnused() {
    static_assert(SourceLocation(0, nullptr).unused_column_ == 0,
                  "Use the otherwise-unused member.");
    return 0;
  }

  // "unused" members are present to minimize future changes in the size of this
  // type.
  std::uint_least32_t line_;
  std::uint_least32_t unused_column_ = 0;
  const char* file_name_;
};

}  // namespace xabsl

// If a function takes an `xabsl::SourceLocation` parameter, pass this as the
// argument.
#define XABSL_LOC \
  ::xabsl::SourceLocation::DoNotInvokeDirectly(__LINE__, __FILE__)

// ABSL_LOC_CURRENT_DEFAULT_ARG
//
// Specifies that a function should use `xabsl::SourceLocation::current()` on
// platforms where it will return useful information, but require explicitly
// passing `XABSL_LOC` on platforms where it would return dummy information.
//
// Usage:
//
//   void MyLog(std::string_view msg,
//              xabsl::SourceLocation loc XABSL_LOC_CURRENT_DEFAULT_ARG) {
//     std::cout << loc.file_name() << "@" << loc.line() << ": " << msg;
//   }
//
#if XABSL_HAVE_SOURCE_LOCATION_CURRENT
#define XABSL_LOC_CURRENT_DEFAULT_ARG = ::xabsl::SourceLocation::current()
#else
#define XABSL_LOC_CURRENT_DEFAULT_ARG
#endif
#endif  // XLS_USE_ABSL_SOURCE_LOCATION

#endif  // XLS_COMMON_SOURCE_LOCATION_H_
