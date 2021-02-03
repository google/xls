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

# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "network_config_proto",
    srcs = ["network_config.proto"],
)

cc_proto_library(
    name = "network_config_cc_proto",
    deps = [
        ":network_config_proto",
    ],
)

cc_library(
    name = "network_config_proto_builder_utils",
    hdrs = ["network_config_proto_builder_utils.h"],
)

cc_library(
    name = "link_config_proto_builder",
    srcs = ["link_config_proto_builder.cc"],
    hdrs = ["link_config_proto_builder.h"],
    deps = [
        ":network_config_cc_proto",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "link_config_proto_builder_test",
    srcs = ["link_config_proto_builder_test.cc"],
    deps = [
        ":link_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "virtual_channel_config_proto_builder",
    srcs = ["virtual_channel_config_proto_builder.cc"],
    hdrs = ["virtual_channel_config_proto_builder.h"],
    deps = [
        ":network_config_cc_proto",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "virtual_channel_config_proto_builder_test",
    srcs = ["virtual_channel_config_proto_builder_test.cc"],
    deps = [
        ":virtual_channel_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "port_config_proto_builder",
    srcs = ["port_config_proto_builder.cc"],
    hdrs = ["port_config_proto_builder.h"],
    deps = [
        ":network_config_cc_proto",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "port_config_proto_builder_test",
    srcs = ["port_config_proto_builder_test.cc"],
    deps = [
        ":port_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "routing_scheme_config_proto_builder",
    srcs = ["routing_scheme_config_proto_builder.cc"],
    hdrs = ["routing_scheme_config_proto_builder.h"],
    deps = [
        ":network_config_cc_proto",
        ":network_config_proto_builder_utils",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "routing_scheme_config_proto_builder_test",
    srcs = ["routing_scheme_config_proto_builder_test.cc"],
    deps = [
        ":routing_scheme_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "arbiter_scheme_config_proto_builder",
    srcs = ["arbiter_scheme_config_proto_builder.cc"],
    hdrs = ["arbiter_scheme_config_proto_builder.h"],
    deps = [
        ":network_config_cc_proto",
        ":network_config_proto_builder_utils",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "arbiter_scheme_config_proto_builder_test",
    srcs = ["arbiter_scheme_config_proto_builder_test.cc"],
    deps = [
        ":arbiter_scheme_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "router_config_proto_builder",
    srcs = ["router_config_proto_builder.cc"],
    hdrs = ["router_config_proto_builder.h"],
    deps = [
        ":arbiter_scheme_config_proto_builder",
        ":network_config_cc_proto",
        ":port_config_proto_builder",
        ":routing_scheme_config_proto_builder",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "router_config_proto_builder_test",
    srcs = ["router_config_proto_builder_test.cc"],
    deps = [
        ":router_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_config_proto_builder",
    srcs = ["network_config_proto_builder.cc"],
    hdrs = ["network_config_proto_builder.h"],
    deps = [
        ":link_config_proto_builder",
        ":network_config_cc_proto",
        ":port_config_proto_builder",
        ":router_config_proto_builder",
        ":virtual_channel_config_proto_builder",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:proto_adaptor_utils",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "network_config_proto_builder_test",
    srcs = ["network_config_proto_builder_test.cc"],
    deps = [
        ":network_config_proto_builder",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "network_config_builder_options_proto",
    srcs = ["network_config_builder_options.proto"],
    deps = [
        ":network_config_proto",
    ],
)

cc_proto_library(
    name = "network_config_builder_options_cc_proto",
    deps = [
        ":network_config_builder_options_proto",
    ],
)

cc_library(
    name = "common_network_config_builder_options_proto_builder_h",
    hdrs = ["common_network_config_builder_options_proto_builder.h"],
    deps = [
        ":network_config_builder_options_cc_proto",
        ":network_config_cc_proto",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "endpoint_options_proto_builder",
    srcs = ["endpoint_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "endpoint_options_proto_builder_test",
    srcs = ["endpoint_options_proto_builder_test.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
        ":endpoint_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "data_options_proto_builder",
    srcs = ["data_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "data_options_proto_builder_test",
    srcs = ["data_options_proto_builder_test.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
        ":data_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flow_control_options_proto_builder",
    srcs = ["flow_control_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "flow_control_options_proto_builder_test",
    srcs = ["flow_control_options_proto_builder_test.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
        ":flow_control_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "link_options_proto_builder",
    srcs = ["link_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "link_options_proto_builder_test",
    srcs = ["link_options_proto_builder_test.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
        ":link_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "virtual_channel_options_proto_builder",
    srcs = ["virtual_channel_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "virtual_channel_options_proto_builder_test",
    srcs = ["virtual_channel_options_proto_builder_test.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
        ":virtual_channel_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "routing_scheme_options_proto_builder",
    srcs = ["routing_scheme_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "routing_scheme_options_proto_builder_test",
    srcs = ["routing_scheme_options_proto_builder_test.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
        ":routing_scheme_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "arbiter_scheme_options_proto_builder",
    srcs = ["arbiter_scheme_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "arbiter_scheme_options_proto_builder_test",
    srcs = ["arbiter_scheme_options_proto_builder_test.cc"],
    deps = [
        ":arbiter_scheme_options_proto_builder",
        ":common_network_config_builder_options_proto_builder_h",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "router_options_proto_builder",
    srcs = ["router_options_proto_builder.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "router_options_proto_builder_test",
    srcs = ["router_options_proto_builder_test.cc"],
    deps = [
        ":common_network_config_builder_options_proto_builder_h",
        ":router_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "custom_network_config_builder_options_proto_builder_h",
    hdrs = ["custom_network_config_builder_options_proto_builder.h"],
    deps = [
        ":network_config_builder_options_cc_proto",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "grid_network_config_options_proto_builder",
    srcs = ["grid_network_config_options_proto_builder.cc"],
    deps = [
        ":custom_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "grid_network_config_options_proto_builder_test",
    srcs = ["grid_network_config_options_proto_builder_test.cc"],
    deps = [
        ":custom_network_config_builder_options_proto_builder_h",
        ":grid_network_config_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "unidirectional_tree_network_config_options_proto_builder",
    srcs = ["unidirectional_tree_network_config_options_proto_builder.cc"],
    deps = [
        ":custom_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "unidirectional_tree_network_config_options_proto_builder_test",
    srcs = ["unidirectional_tree_network_config_options_proto_builder_test.cc"],
    deps = [
        ":custom_network_config_builder_options_proto_builder_h",
        ":unidirectional_tree_network_config_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bidirectional_tree_network_config_options_proto_builder",
    srcs = ["bidirectional_tree_network_config_options_proto_builder.cc"],
    deps = [
        ":custom_network_config_builder_options_proto_builder_h",
    ],
)

cc_test(
    name = "bidirectional_tree_network_config_options_proto_builder_test",
    srcs = ["bidirectional_tree_network_config_options_proto_builder_test.cc"],
    deps = [
        ":bidirectional_tree_network_config_options_proto_builder",
        ":custom_network_config_builder_options_proto_builder_h",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "custom_network_config_builder_options_proto_builder",
    deps = [
        ":bidirectional_tree_network_config_options_proto_builder",
        ":grid_network_config_options_proto_builder",
        ":unidirectional_tree_network_config_options_proto_builder",
    ],
)
