// Copyright 2025 The XLS Authors
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

#ifndef XLS_COMMON_STATUS_STATUS_OR_REF_H_
#define XLS_COMMON_STATUS_STATUS_OR_REF_H_

#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_builder.h"

namespace xls {

// Helper to make it easier to return references through status-ors. This
// implements the main StatusOr functions enough to be used by the normal
// StatusOr macros.
template <typename T>
class [[nodiscard]] StatusOrRef {
  using InnerStatusOr =
#ifdef ABSL_NONNULL
      absl::StatusOr<T* const ABSL_NONNULL>;
#else
      absl::StatusOr<T* const>;
#endif

 public:
  typedef T value_type;
  // Constructs with an error of kUnknown
  explicit StatusOrRef() : value_() {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  StatusOrRef(T& value ABSL_ATTRIBUTE_LIFETIME_BOUND) : value_(&value) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  StatusOrRef(InnerStatusOr value) : value_(std::move(value)) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  StatusOrRef(absl::Status value) : value_(std::move(value)) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  StatusOrRef(xabsl::StatusBuilder value) : value_(std::move(value)) {}

  StatusOrRef& operator=(
      const StatusOrRef<T>& x ABSL_ATTRIBUTE_LIFETIME_BOUND) = default;
  StatusOrRef& operator=(StatusOrRef<T>&& x ABSL_ATTRIBUTE_LIFETIME_BOUND) =
      default;
  StatusOrRef(const StatusOrRef<T>& x ABSL_ATTRIBUTE_LIFETIME_BOUND) = default;
  StatusOrRef(StatusOrRef<T>&& x ABSL_ATTRIBUTE_LIFETIME_BOUND) = default;

  // Conversion constructors
  template <typename U>
    requires(std::is_base_of_v<T, U>)
  StatusOrRef(StatusOrRef<U>&& other ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : value_(std::move(other.value_)) {}
  template <typename U>
    requires(std::is_base_of_v<T, U>)
  StatusOrRef& operator=(StatusOrRef<U>&& other ABSL_ATTRIBUTE_LIFETIME_BOUND) {
    value_ = std::move(other.value_);
    return *this;
  }
  // conversion to base status-or.
  operator InnerStatusOr() && { return std::move(value_); }
  operator InnerStatusOr() & { return value_; }
  operator InnerStatusOr() const& { return value_; }

  bool ok() const { return value_.ok(); }

  ABSL_MUST_USE_RESULT const absl::Status& status() const& {
    return value_.status();
  }
  ABSL_MUST_USE_RESULT absl::Status status() && { return value_.status(); }

  T& operator*() { return *value_.value(); }
  const T& operator*() const { return *value_.value(); }

  T* operator->() { return value_.value(); }
  const T* operator->() const { return value_.value(); }

  T& value() & { return *value_.value(); }
  const T& value() const& { return *value_.value(); }
  T& value() && { return *std::move(value_).value(); }

  void IgnoreError() const { value_.IgnoreError(); }

 private:
  InnerStatusOr value_;
};

template <typename T>
bool operator==(const StatusOrRef<T>& l, const StatusOrRef<T>& r) {
  if (l.ok() && r.ok()) {
    return l.value() == r.value();
  }
  return l.status() == r.status();
}
template <typename T>
bool operator!=(const StatusOrRef<T>& l, const StatusOrRef<T>& r) {
  return !(l == r);
}

// Make sure status matchers can unwrap the status.
template <typename T>
inline const absl::Status& GetStatus(const StatusOrRef<T>& v) {
  return v.status();
}

// TODO(allight): Add stringify and <<

}  // namespace xls

#endif  // XLS_COMMON_STATUS_STATUS_OR_REF_H_
