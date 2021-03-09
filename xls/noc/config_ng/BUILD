# Copyright 2021 The XLS Authors
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
    name = "topology_options_network_config_builder_proto",
    srcs = ["topology_options_network_config_builder.proto"],
)

cc_proto_library(
    name = "topology_options_network_config_builder_cc_proto",
    deps = [
        ":topology_options_network_config_builder_proto",
    ],
)

proto_library(
    name = "component_configuration_options_network_config_builder_proto",
    srcs = ["component_configuration_options_network_config_builder.proto"],
    deps = ["//xls/noc/config:network_config_proto"],
)

cc_proto_library(
    name = "component_configuration_options_network_config_builder_cc_proto",
    deps = [
        ":component_configuration_options_network_config_builder_proto",
    ],
)

proto_library(
    name = "routing_algorithm_options_network_config_builder_proto",
    srcs = ["routing_algorithm_options_network_config_builder.proto"],
)

cc_proto_library(
    name = "routing_algorithm_options_network_config_builder_cc_proto",
    deps = [
        ":routing_algorithm_options_network_config_builder_proto",
    ],
)

proto_library(
    name = "network_config_builder_options_proto",
    srcs = ["network_config_builder_options.proto"],
    deps = [
        ":component_configuration_options_network_config_builder_proto",
        ":routing_algorithm_options_network_config_builder_proto",
        ":topology_options_network_config_builder_proto",
    ],
)

cc_proto_library(
    name = "network_config_builder_options_cc_proto",
    deps = [
        ":network_config_builder_options_proto",
    ],
)

cc_library(
    name = "topology_endpoint_options_proto_builder",
    srcs = ["topology_endpoint_options_proto_builder.cc"],
    hdrs = ["topology_endpoint_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:integral_types",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "topology_endpoint_options_proto_builder_test",
    srcs = ["topology_endpoint_options_proto_builder_test.cc"],
    deps = [
        ":topology_endpoint_options_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_component_port",
    srcs = ["network_component_port.cc"],
    hdrs = ["network_component_port.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "network_component_port_test",
    srcs = ["network_component_port_test.cc"],
    deps = [
        ":network_component",
        ":network_component_port",
        ":network_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_component",
    srcs = ["network_component.cc"],
    hdrs = ["network_component.h"],
    deps = [
        ":network_component_port",
        "//xls/common:integral_types",
        "//xls/common:iterator_range",
        "//xls/common/logging",
        "//xls/ir:unwrapping_iterator",
    ],
)

cc_test(
    name = "network_component_test",
    srcs = ["network_component_test.cc"],
    deps = [
        ":network_component",
        ":network_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_connection",
    srcs = ["network_connection.cc"],
    hdrs = ["network_connection.h"],
    deps = [
        ":network_component_port",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "network_connection_test",
    srcs = ["network_connection_test.cc"],
    deps = [
        ":network_component",
        ":network_component_port",
        ":network_connection",
        ":network_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_view",
    srcs = ["network_view.cc"],
    hdrs = ["network_view.h"],
    deps = [
        ":network_component",
        ":network_connection",
        "@com_google_absl//absl/memory",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "network_view_test",
    srcs = ["network_view_test.cc"],
    deps = [
        ":network_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fake_network_component",
    testonly = True,
    hdrs = ["fake_network_component.h"],
    deps = [
        ":network_component",
    ],
)

cc_library(
    name = "network_component_port_utils",
    srcs = ["network_component_port_utils.cc"],
    hdrs = ["network_component_port_utils.h"],
    deps = [
        ":network_component_port",
        ":network_connection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
    ],
)

cc_test(
    name = "network_component_port_utils_test",
    srcs = ["network_component_port_utils_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_component_port",
        ":network_component_port_utils",
        ":network_view",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_component_utils",
    srcs = ["network_component_utils.cc"],
    hdrs = ["network_component_utils.h"],
    deps = [
        ":network_component",
        ":network_component_port_utils",
        ":network_connection",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "network_component_utils_test",
    srcs = ["network_component_utils_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_component_utils",
        ":network_view",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_connection_utils",
    srcs = ["network_connection_utils.cc"],
    hdrs = ["network_connection_utils.h"],
    deps = [
        ":network_component_port",
        ":network_connection",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "network_connection_utils_test",
    srcs = ["network_connection_utils_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_component_port",
        ":network_connection_utils",
        ":network_view",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_view_utils",
    srcs = ["network_view_utils.cc"],
    hdrs = ["network_view_utils.h"],
    deps = [
        ":network_component_utils",
        ":network_connection_utils",
        ":network_view",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "network_view_utils_test",
    srcs = ["network_view_utils_test.cc"],
    deps = [
        ":network_view",
        ":network_view_utils",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
