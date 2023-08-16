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

proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
)

cc_proto_library(
    name = "cc_config_proto",
    deps = [":config_proto"],
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
    name = "config",
    srcs = [
        "config.cc",
    ],
    hdrs = [
        "config.h",
    ],
    deps = [
        ":cc_config_proto",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "config_test",
    srcs = [
        "config_test.cc",
    ],
    deps = [
        ":config",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
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

cc_library(
    name = "iregister",
    hdrs = [
        "iregister.h",
    ],
    deps = [
        ":ichannel",
    ],
)

cc_library(
    name = "iregister_stub",
    srcs = [
        "iregister_stub.cc",
    ],
    hdrs = [
        "iregister_stub.h",
    ],
    deps = [
        ":iregister",
        "//xls/common/logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "iregister_stub_test",
    size = "small",
    srcs = [
        "iregister_stub_test.cc",
    ],
    deps = [
        ":iregister_stub",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_flags",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "iregister_mock",
    hdrs = [
        "iregister_mock.h",
    ],
    deps = [
        ":byteops",
        ":iregister",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "iregister_mock_test",
    size = "small",
    srcs = [
        "iregister_mock_test.cc",
    ],
    deps = [
        ":iregister_mock",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "istream",
    hdrs = [
        "istream.h",
    ],
    deps = [
        ":ichannel",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "istream_stub",
    srcs = [
        "istream_stub.cc",
    ],
    hdrs = [
        "istream_stub.h",
    ],
    deps = [
        ":istream",
        "//xls/common/logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "istream_stub_test",
    size = "small",
    srcs = [
        "istream_stub_test.cc",
    ],
    deps = [
        ":istream_stub",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging:log_flags",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "istream_mock",
    hdrs = [
        "istream_mock.h",
    ],
    deps = [
        ":iregister_mock",
        ":istream",
        "//xls/common/logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "istream_mock_test",
    size = "small",
    srcs = [
        "istream_mock_test.cc",
    ],
    deps = [
        ":istream_mock",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ichannelmanager",
    hdrs = [
        "ichannelmanager.h",
    ],
    deps = [
        ":ichannel",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "ichannelmanager_stub_test",
    srcs = [
        "ichannelmanager_stub.cc",
        "ichannelmanager_stub.h",
        "ichannelmanager_stub_test.cc",
    ],
    deps = [
        ":ichannel",
        ":ichannelmanager",
        ":istream_stub",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/logging:log_flags",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "iaxistreamlike",
    hdrs = [
        "iaxistreamlike.h",
    ],
    deps = [
        ":istream",
    ],
)

cc_library(
    name = "iaxistreamlike_stub",
    srcs = [
        "iaxistreamlike_stub.cc",
    ],
    hdrs = [
        "iaxistreamlike_stub.h",
    ],
    deps = [
        ":iaxistreamlike",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "iaxistreamlike_stub_test",
    srcs = [
        "iaxistreamlike_stub_test.cc",
    ],
    deps = [
        ":iaxistreamlike_stub",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "iperipheral",
    hdrs = [
        "iperipheral.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "iperipheral_stub",
    srcs = [
        "iperipheral_stub.cc",
    ],
    hdrs = [
        "iperipheral_stub.h",
    ],
    deps = [
        ":iperipheral",
        "//xls/common/logging",
        "@com_google_absl//absl/strings:str_format",
    ],
)
