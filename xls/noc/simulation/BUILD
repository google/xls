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
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "common_test",
    srcs = ["common_test.cc"],
    deps = [
        ":common",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_graph",
    srcs = ["network_graph.cc"],
    hdrs = ["network_graph.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "network_graph_test",
    srcs = ["network_graph_test.cc"],
    deps = [
        ":common",
        ":network_graph",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parameters",
    hdrs = ["parameters.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/noc/config:network_config_cc_proto",
    ],
)

cc_test(
    name = "parameters_test",
    srcs = ["parameters_test.cc"],
    deps = [
        ":parameters",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "network_graph_builder",
    srcs = ["network_graph_builder.cc"],
    hdrs = ["network_graph_builder.h"],
    deps = [
        ":common",
        ":network_graph",
        ":parameters",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/noc/config:network_config_cc_proto",
    ],
)

cc_test(
    name = "network_graph_builder_test",
    srcs = ["network_graph_builder_test.cc"],
    deps = [
        ":network_graph_builder",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "indexer",
    srcs = ["indexer.cc"],
    hdrs = ["indexer.h"],
    deps = [
        ":common",
        ":parameters",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "indexer_test",
    srcs = ["indexer_test.cc"],
    deps = [
        ":indexer",
        ":network_graph_builder",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "global_routing_table",
    srcs = ["global_routing_table.cc"],
    hdrs = ["global_routing_table.h"],
    deps = [
        ":common",
        ":indexer",
        ":network_graph",
        ":parameters",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/noc/config:network_config_cc_proto",
    ],
)

cc_test(
    name = "global_routing_table_test",
    srcs = ["global_routing_table_test.cc"],
    deps = [
        ":global_routing_table",
        ":network_graph_builder",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sim_objects",
    srcs = ["sim_objects.cc"],
    hdrs = ["sim_objects.h"],
    deps = [
        ":common",
        ":global_routing_table",
        ":network_graph",
        ":parameters",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/noc/config:network_config_cc_proto",
    ],
)

cc_test(
    name = "sim_objects_test",
    srcs = ["sim_objects_test.cc"],
    deps = [
        ":network_graph_builder",
        ":sim_objects",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest_main",
    ],
)
