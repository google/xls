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

#include "absl/status/statusor.h"
namespace xls {

// Exception to throw when we return a non-ok status.
class StatusNotOk : public std::exception {
 public:
  StatusNotOk(absl::Status&& status)
      : status_(std::move(status)), what_(status_.ToString()) {}
  StatusNotOk(const absl::Status& status)
      : status_(status), what_(status_.ToString()) {}
  const absl::Status& status() const { return status_; }
  const char* what() const noexcept override { return what_.c_str(); }

 private:
  absl::Status status_;
  std::string what_;
};

// Stub ImportStatusModule.
void ImportStatusModule() {}

}  // namespace xls

namespace pybind11 {
namespace detail {

template <>
struct type_caster<absl::Status> {
 public:
   static constexpr auto name = _("Status");

  // Conversion part 2 (C++ -> Python)
  static handle cast(const absl::Status* src, return_value_policy policy,
                     handle parent, bool throw_exception = true) {
    if (!src) return none().release();
    return cast_impl<const absl::Status&>(*src, policy, parent,
                                          throw_exception);
  }

  static handle cast(const absl::Status& src, return_value_policy policy,
                     handle parent, bool throw_exception = true) {
    return cast_impl<const absl::Status&>(src, policy, parent,
    throw_exception);
  }

  static handle cast(absl::Status&& src, return_value_policy policy,
                     handle parent, bool throw_exception = true) {
    return cast_impl<absl::Status&&>(std::move(src), policy, parent,
                                     throw_exception);
  }

 private:
  template <typename CType>
  static handle cast_impl(CType src, return_value_policy policy, handle
  parent,
                          bool throw_exception) {
    if (!throw_exception) {
      // Use the built-in/standard pybind11 caster.
      return type_caster_base<absl::Status>::cast(std::forward<CType>(src),
                                                  policy, parent);
    } else if (!src.ok()) {
      // Convert a non-ok status into an exception.
      throw xls::StatusNotOk(std::forward<CType>(src));
    } else {
      // Return none for an ok status.
      return none().release();
    }
  }
};

template <typename PayloadType>
struct type_caster<absl::StatusOr<PayloadType>> {
 public:
  using PayloadCaster = make_caster<PayloadType>;
  using StatusCaster = make_caster<absl::Status>;
  static constexpr auto name = _("StatusOr[") + PayloadCaster::name + _("]");

  // Conversion part 2 (C++ -> Python).
  static handle cast(absl::StatusOr<PayloadType>&& src,
                     return_value_policy policy, handle parent,
                     bool throw_exception = true) {
    if (src.ok()) {
      // Convert and return the payload.
      return PayloadCaster::cast(std::forward<PayloadType>(*src), policy,
                                 parent);
    } else {
      // Convert and return the error.
      return StatusCaster::cast(std::move(src.status()),
                                return_value_policy::move, parent,
                                throw_exception);
    }
  }
};

}  // namespace detail
}  // namespace pybind11
