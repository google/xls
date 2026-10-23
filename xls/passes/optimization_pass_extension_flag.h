// Copyright 2026 The XLS Authors
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

#ifndef XLS_PASSES_OPTIMIZATION_PASS_EXTENSION_FLAG_H_
#define XLS_PASSES_OPTIMIZATION_PASS_EXTENSION_FLAG_H_

#include <optional>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/types/optional_ref.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/tools/opt_flags.pb.h"
#include "xls/tools/opt_flags_handler_registry.h"

namespace xls {

namespace internal {

// Workaround for LINT checks on safe use of GetFlag.
template <typename T>
T InternalGetFlag(const absl::Flag<T>& flag) {
  return absl::GetFlag(flag);  // NOLINT
}

template <typename T>
class OptimizationPassExtensionFlag : public OptimizationPassOptionExtension {
 public:
  explicit OptimizationPassExtensionFlag(std::string_view name, T value)
      : name_(name), value_(std::move(value)) {}
  const T& value() const { return value_; }
  std::string_view extension_name() const override { return name_; }

  template <typename ProtoExt>
  static OptFlagsParser CreateParser(const absl::Flag<T>& flag,
                                     const ProtoExt& proto_ext) {
    return [&flag, &proto_ext](OptFlagsProto& proto) -> absl::Status {
      proto.SetExtension(proto_ext, InternalGetFlag(flag));
      return absl::OkStatus();
    };
  }

  template <typename ProtoExt>
  static OptFlagsHandler CreateHandler(const absl::Flag<T>& flag,
                                       const ProtoExt& proto_ext) {
    return
        [&flag, &proto_ext](const OptFlagsProto& proto,
                            OptimizationPassOptions& options) -> absl::Status {
          XLS_RETURN_IF_ERROR(options.AddExtension(
              std::make_unique<OptimizationPassExtensionFlag<T>>(
                  flag.Name(), proto.GetExtension(proto_ext))))
              << "Unable to register extension flag " << flag.Name();
          return absl::OkStatus();
        };
  }

 private:
  std::string_view name_;
  T value_;
};
}  // namespace internal

// Helper to read out the actually configured value of an extension flag (either
// from the flag itself or supplied through a provided protobuf).
//
// This should always be used for pass-specific flags.
//
// Using absl::GetFlag (NOLINT) on these flags will not work as expected.
template <typename T>
absl::optional_ref<const T> GetConfiguredExtensionFlag(
    const OptimizationPassOptions& options, const absl::Flag<T>& flag) {
  absl::optional_ref<const internal::OptimizationPassExtensionFlag<T>> v =
      options.GetExtension<internal::OptimizationPassExtensionFlag<T>>(
          flag.Name());
  if (!v.has_value()) {
    return std::nullopt;
  }
  return v->value();
}

// Helper to setup a pass-specific extension flag.
//
// USAGE:
// In internal_opt_flags.proto
//   ...
//   extend xls.OptFlagsProto {
//     bool my_pass_enabled = 20002;
//   }
//
// In your pass:
//   ABSL_FLAG(bool, my_pass_enabled, false, "Enable my pass.");
//   XLS_REGISTER_OPT_PASS_EXT_FLAG(FLAGS_my_pass_enabled,
//                                  xls::opt_ext_flags::my_pass_enabled);
//
// In your pass code:
//   ...
//   bool enabled =
//       GetConfiguredExtensionFlag(options, FLAGS_my_pass_enabled).value_or(
//           false);
//   ...
//
// Currently only works with simple types which protobuf natively supports
// (bool, int, float, string, etc.).
//
// TODO(allight): Might be nice to allow for some way to handle enums.
#define XLS_REGISTER_OPT_PASS_EXT_FLAG(flag, proto_ext) \
  XLS_REGISTER_OPT_FLAGS_HANDLER(                       \
      RegisterFlagForExtension_##flag,                  \
      ::xls::internal::OptimizationPassExtensionFlag<   \
          decltype(::xls::internal::InternalGetFlag(    \
              flag))>::CreateHandler(flag, proto_ext),  \
      ::xls::internal::OptimizationPassExtensionFlag<   \
          decltype(::xls::internal::InternalGetFlag(    \
              flag))>::CreateParser(flag, proto_ext))

}  // namespace xls

#endif  // XLS_PASSES_OPTIMIZATION_PASS_EXTENSION_FLAG_H_
