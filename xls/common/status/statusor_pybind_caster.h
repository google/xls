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

#ifndef XLS_COMMON_STATUS_STATUSOR_PYBIND_CASTER_H_
#define XLS_COMMON_STATUS_STATUSOR_PYBIND_CASTER_H_

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include "absl/status/status.h"
#include "xls/common/status/statusor.h"
#include "xls/common/status/statusor_pybind_caster.inc"

namespace pybind11 {
namespace detail {

// Convert an xabsl::StatusOr.
template <typename PayloadType>
struct type_caster<xabsl::StatusOr<PayloadType>> {
 public:
  using PayloadCaster = make_caster<PayloadType>;
  using StatusCaster = make_caster<absl::Status>;
  static constexpr auto name = _("StatusOr[") + PayloadCaster::name + _("]");

  // Conversion part 2 (C++ -> Python).
  static handle cast(xabsl::StatusOr<PayloadType>&& src,
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

#endif  // XLS_COMMON_STATUS_STATUSOR_PYBIND_CASTER_H_
