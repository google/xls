# Copyright 2020 The XLS Authors
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
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    deps = [
        "//xls/common/logging",
    ],
)

cc_library(
    name = "iconnection",
    hdrs = [
        "iconnection.h",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "iactive",
    hdrs = [
        "iactive.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "iactive_test",
    srcs = [
        "iactive_stub.cc",
        "iactive_stub.h",
        "iactive_test.cc",
    ],
    deps = [
        ":iactive",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "iirq",
    hdrs = [
        "iirq.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "iirq_test",
    srcs = [
        "iirq_stub.cc",
        "iirq_stub.h",
        "iirq_test.cc",
    ],
    deps = [
        ":iirq",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "byteops",
    hdrs = [
        "byteops.h",
    ],
    deps = [
        "//xls/common/logging",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "byteops_test",
    size = "small",
    srcs = [
        "byteops_test.cc",
    ],
    deps = [
        ":byteops",
        "//xls/common/status:matchers",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ichannel",
    hdrs = [
        "ichannel.h",
    ],
    deps = [
        ":common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
