# Copyright 2026 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "busperf_yaml_generator",
    srcs = ["busperf_yaml_generator.cc"],
    hdrs = ["busperf_yaml_generator.h"],
    deps = [
        "//xls/codegen:module_signature_cc_proto",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "busperf_yaml_generator_test",
    srcs = ["busperf_yaml_generator_test.cc"],
    deps = [
        ":busperf_yaml_generator",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest",
        "@protobuf",
    ],
)

cc_binary(
    name = "xls_sig_to_busperf",
    srcs = ["xls_sig_to_busperf_main.cc"],
    visibility = ["//xls/experimental/busperf:__subpackages__"],
    deps = [
        ":busperf_yaml_generator",
        "//xls/codegen:module_signature_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)
