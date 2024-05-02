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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "units",
    hdrs = ["units.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "units_test",
    srcs = ["units_test.cc"],
    deps = [
        ":units",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "network_graph",
    srcs = ["network_graph.cc"],
    hdrs = ["network_graph.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parameters",
    hdrs = ["parameters.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/noc/config:network_config_cc_proto",
    ],
)

cc_test(
    name = "parameters_test",
    srcs = ["parameters_test.cc"],
    deps = [
        ":parameters",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//xls/noc/config:network_config_cc_proto",
    ],
)

cc_test(
    name = "network_graph_builder_test",
    srcs = ["network_graph_builder_test.cc"],
    deps = [
        ":network_graph_builder",
        "@com_google_absl//absl/log",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "sample_network_graphs",
    srcs = ["sample_network_graphs.cc"],
    hdrs = ["sample_network_graphs.h"],
    deps = [
        ":common",
        ":network_graph",
        ":network_graph_builder",
        ":parameters",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "indexer_test",
    srcs = ["indexer_test.cc"],
    deps = [
        ":indexer",
        "@com_google_absl//absl/status",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
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
        ":sample_network_graphs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "flit",
    hdrs = ["flit.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:status_macros",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
    ],
)

cc_library(
    name = "sim_objects",
    srcs = ["sim_objects.cc"],
    hdrs = ["sim_objects.h"],
    deps = [
        ":common",
        ":flit",
        ":global_routing_table",
        ":network_graph",
        ":parameters",
        ":simulator_shims",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:ret_check",
        "//xls/ir:bits",
        "//xls/noc/config:network_config_cc_proto",
    ],
)

cc_test(
    name = "sim_objects_test",
    srcs = ["sim_objects_test.cc"],
    deps = [
        ":flit",
        ":network_graph_builder",
        ":sample_network_graphs",
        ":sim_objects",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/noc/config:network_config_cc_proto",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "packetizer",
    srcs = ["packetizer.cc"],
    hdrs = ["packetizer.h"],
    deps = [
        ":common",
        ":sim_objects",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
    ],
)

cc_test(
    name = "packetizer_test",
    srcs = ["packetizer_test.cc"],
    deps = [
        ":packetizer",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "sim_packets_test",
    srcs = ["sim_packets_test.cc"],
    deps = [
        ":packetizer",
        ":sample_network_graphs",
        ":sim_objects",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "traffic_description",
    hdrs = ["traffic_description.h"],
    deps = [
        ":common",
        ":units",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "traffic_description_test",
    srcs = ["traffic_description_test.cc"],
    deps = [
        ":traffic_description",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "random_number_interface",
    hdrs = ["random_number_interface.h"],
    deps = [
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_test(
    name = "random_number_interface_test",
    srcs = ["random_number_interface_test.cc"],
    deps = [
        ":random_number_interface",
        "//xls/common:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "traffic_models",
    srcs = ["traffic_models.cc"],
    hdrs = ["traffic_models.h"],
    deps = [
        ":common",
        ":packetizer",
        ":random_number_interface",
        ":units",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "traffic_models_test",
    srcs = ["traffic_models_test.cc"],
    deps = [
        ":traffic_models",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "simulator_shims",
    hdrs = ["simulator_shims.h"],
    deps = [
        ":common",
        ":flit",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "simulator_to_traffic_injector_shim",
    hdrs = ["simulator_to_traffic_injector_shim.h"],
    deps = [
        ":common",
        ":noc_traffic_injector",
        ":sim_objects",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "simulator_to_link_monitor_shim",
    srcs = ["simulator_to_link_monitor_service_shim.cc"],
    hdrs = ["simulator_to_link_monitor_service_shim.h"],
    deps = [
        ":common",
        ":sim_objects",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "noc_traffic_injector",
    srcs = ["noc_traffic_injector.cc"],
    hdrs = ["noc_traffic_injector.h"],
    deps = [
        ":common",
        ":flit",
        ":indexer",
        ":network_graph",
        ":packetizer",
        ":random_number_interface",
        ":simulator_shims",
        ":traffic_description",
        ":traffic_models",
        ":units",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "noc_traffic_injector_test",
    srcs = ["noc_traffic_injector_test.cc"],
    deps = [
        ":noc_traffic_injector",
        ":packetizer",
        ":sample_network_graphs",
        ":traffic_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "sim_traffic_test",
    srcs = ["sim_traffic_test.cc"],
    deps = [
        ":noc_traffic_injector",
        ":sample_network_graphs",
        ":sim_objects",
        ":simulator_to_link_monitor_shim",
        ":simulator_to_traffic_injector_shim",
        ":traffic_description",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)
