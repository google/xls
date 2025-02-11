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

# Load proto_library
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
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
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "topology_endpoint_options_proto_builder_test",
    srcs = ["topology_endpoint_options_proto_builder_test.cc"],
    deps = [
        ":topology_endpoint_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "dimension_order_entry_options_proto_builder",
    srcs = ["dimension_order_entry_options_proto_builder.cc"],
    hdrs = ["dimension_order_entry_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "dimension_order_entry_options_proto_builder_test",
    srcs = ["dimension_order_entry_options_proto_builder_test.cc"],
    deps = [
        ":dimension_order_entry_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "dimension_order_options_proto_builder",
    srcs = ["dimension_order_options_proto_builder.cc"],
    hdrs = ["dimension_order_options_proto_builder.h"],
    deps = [
        ":dimension_order_entry_options_proto_builder",
        ":topology_endpoint_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "dimension_order_options_proto_builder_test",
    srcs = ["dimension_order_options_proto_builder_test.cc"],
    deps = [
        ":dimension_order_entry_options_proto_builder",
        ":dimension_order_options_proto_builder",
        ":topology_endpoint_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "unidirectional_butterfly_options_proto_builder",
    srcs = ["unidirectional_butterfly_options_proto_builder.cc"],
    hdrs = ["unidirectional_butterfly_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "unidirectional_butterfly_options_proto_builder_test",
    srcs = ["unidirectional_butterfly_options_proto_builder_test.cc"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        ":unidirectional_butterfly_options_proto_builder",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "bidirectional_butterfly_options_proto_builder",
    srcs = ["bidirectional_butterfly_options_proto_builder.cc"],
    hdrs = ["bidirectional_butterfly_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "bidirectional_butterfly_options_proto_builder_test",
    srcs = ["bidirectional_butterfly_options_proto_builder_test.cc"],
    deps = [
        ":bidirectional_butterfly_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "butterfly_options_proto_builder",
    srcs = ["butterfly_options_proto_builder.cc"],
    hdrs = ["butterfly_options_proto_builder.h"],
    deps = [
        ":bidirectional_butterfly_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        ":unidirectional_butterfly_options_proto_builder",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "butterfly_options_proto_builder_test",
    srcs = ["butterfly_options_proto_builder_test.cc"],
    deps = [
        ":bidirectional_butterfly_options_proto_builder",
        ":butterfly_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        ":unidirectional_butterfly_options_proto_builder",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "level_options_proto_builder",
    srcs = ["level_options_proto_builder.cc"],
    hdrs = ["level_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "level_options_proto_builder_test",
    srcs = ["level_options_proto_builder_test.cc"],
    deps = [
        ":level_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "unidirectional_tree_options_proto_builder",
    srcs = ["unidirectional_tree_options_proto_builder.cc"],
    hdrs = ["unidirectional_tree_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "unidirectional_tree_options_proto_builder_test",
    srcs = ["unidirectional_tree_options_proto_builder_test.cc"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        ":unidirectional_tree_options_proto_builder",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "bidirectional_tree_options_proto_builder",
    srcs = ["bidirectional_tree_options_proto_builder.cc"],
    hdrs = ["bidirectional_tree_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "bidirectional_tree_options_proto_builder_test",
    srcs = ["bidirectional_tree_options_proto_builder_test.cc"],
    deps = [
        ":bidirectional_tree_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "tree_options_proto_builder",
    srcs = ["tree_options_proto_builder.cc"],
    hdrs = ["tree_options_proto_builder.h"],
    deps = [
        ":bidirectional_tree_options_proto_builder",
        ":level_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        ":unidirectional_tree_options_proto_builder",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tree_options_proto_builder_test",
    srcs = ["tree_options_proto_builder_test.cc"],
    deps = [
        ":bidirectional_tree_options_proto_builder",
        ":level_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        ":tree_options_proto_builder",
        ":unidirectional_tree_options_proto_builder",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "fully_connected_options_proto_builder",
    srcs = ["fully_connected_options_proto_builder.cc"],
    hdrs = ["fully_connected_options_proto_builder.h"],
    deps = [
        ":topology_endpoint_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "fully_connected_options_proto_builder_test",
    srcs = ["fully_connected_options_proto_builder_test.cc"],
    deps = [
        ":fully_connected_options_proto_builder",
        ":topology_endpoint_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "coordinate_options_proto_builder",
    srcs = ["coordinate_options_proto_builder.cc"],
    hdrs = ["coordinate_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "coordinate_options_proto_builder_test",
    srcs = ["coordinate_options_proto_builder_test.cc"],
    deps = [
        ":coordinate_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "node_options_proto_builder",
    srcs = ["node_options_proto_builder.cc"],
    hdrs = ["node_options_proto_builder.h"],
    deps = [
        ":coordinate_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:proto_adaptor_utils",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "node_options_proto_builder_test",
    srcs = ["node_options_proto_builder_test.cc"],
    deps = [
        ":coordinate_options_proto_builder",
        ":node_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "channel_options_proto_builder",
    srcs = ["channel_options_proto_builder.cc"],
    hdrs = ["channel_options_proto_builder.h"],
    deps = [
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:proto_adaptor_utils",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "channel_options_proto_builder_test",
    srcs = ["channel_options_proto_builder_test.cc"],
    deps = [
        ":channel_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "custom_topology_options_proto_builder",
    srcs = ["custom_topology_options_proto_builder.cc"],
    hdrs = ["custom_topology_options_proto_builder.h"],
    deps = [
        ":channel_options_proto_builder",
        ":node_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "custom_topology_options_proto_builder_test",
    srcs = ["custom_topology_options_proto_builder_test.cc"],
    deps = [
        ":channel_options_proto_builder",
        ":custom_topology_options_proto_builder",
        ":node_options_proto_builder",
        ":topology_options_network_config_builder_cc_proto",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "network_component_port",
    srcs = ["network_component_port.cc"],
    hdrs = ["network_component_port.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "network_component_port_test",
    srcs = ["network_component_port_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_component_port",
        ":network_view",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "network_component",
    srcs = ["network_component.cc"],
    hdrs = ["network_component.h"],
    deps = [
        ":network_component_port",
        "//xls/common:iterator_range",
        "//xls/ir:unwrapping_iterator",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "network_component_test",
    srcs = ["network_component_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_component_port",
        ":network_view",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "network_connection",
    srcs = ["network_connection.cc"],
    hdrs = ["network_connection.h"],
    deps = [
        ":network_component_port",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "network_connection_test",
    srcs = ["network_connection_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_component_port",
        ":network_connection",
        ":network_view",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "network_view",
    srcs = ["network_view.cc"],
    hdrs = ["network_view.h"],
    deps = [
        ":network_component",
        ":network_connection",
        "//xls/common:iterator_range",
        "//xls/ir:unwrapping_iterator",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "network_view_test",
    srcs = ["network_view_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_connection",
        ":network_view",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "fake_network_component",
    testonly = True,
    hdrs = ["fake_network_component.h"],
    deps = [
        ":network_component",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "network_component_port_utils",
    srcs = ["network_component_port_utils.cc"],
    hdrs = ["network_component_port_utils.h"],
    deps = [
        ":network_component_port",
        ":network_connection",
        "//xls/common/status:ret_check",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@googletest//:gtest",
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
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "network_component_utils_test",
    srcs = ["network_component_utils_test.cc"],
    deps = [
        ":fake_network_component",
        ":network_component_port",
        ":network_component_utils",
        ":network_view",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@googletest//:gtest",
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
        ":network_component",
        ":network_component_port",
        ":network_connection_utils",
        ":network_view",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
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
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "network_view_utils_test",
    srcs = ["network_view_utils_test.cc"],
    deps = [
        ":network_view",
        ":network_view_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "arbitrary_base_number",
    srcs = ["arbitrary_base_number.cc"],
    hdrs = ["arbitrary_base_number.h"],
    deps = [
        "//xls/ir:bits",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "arbitrary_base_number_test",
    srcs = ["arbitrary_base_number_test.cc"],
    deps = [
        ":arbitrary_base_number",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "dimension_bounds",
    srcs = ["dimension_bounds.cc"],
    hdrs = ["dimension_bounds.h"],
    deps = [
        "//xls/ir:bits",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dimension_bounds_test",
    srcs = ["dimension_bounds_test.cc"],
    deps = [
        ":dimension_bounds",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "coordinate",
    srcs = ["coordinate.cc"],
    hdrs = ["coordinate.h"],
    deps = [
        ":dimension_bounds",
        "//xls/data_structures:inline_bitmap",
        "//xls/ir:bits",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "coordinate_test",
    srcs = ["coordinate_test.cc"],
    deps = [
        ":coordinate",
        ":dimension_bounds",
        "//xls/common:xls_gunit_main",
        "//xls/data_structures:inline_bitmap",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "flattened_multi_dimensional_array",
    hdrs = [
        "flattened_multi_dimensional_array.h",
    ],
    deps = [
        ":coordinate",
        ":dimension_bounds",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "flattened_multi_dimensional_array_test",
    srcs = ["flattened_multi_dimensional_array_test.cc"],
    deps = [
        ":coordinate",
        ":dimension_bounds",
        ":flattened_multi_dimensional_array",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "network_topology_component",
    srcs = ["network_topology_component.cc"],
    hdrs = [
        "network_component_visitor_abstract.h",
        "network_topology_component.h",
    ],
    deps = [
        ":coordinate",
        ":network_component",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "network_topology_component_test",
    srcs = ["network_topology_component_test.cc"],
    deps = [
        ":coordinate",
        ":network_topology_component",
        ":network_topology_view",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "network_topology_view",
    srcs = ["network_topology_view.cc"],
    hdrs = ["network_topology_view.h"],
    deps = [
        ":network_component",
        ":network_component_port",
        ":network_topology_component",
        ":network_view",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "network_topology_view_test",
    srcs = ["network_topology_view_test.cc"],
    deps = [
        ":network_component",
        ":network_topology_view",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@googletest//:gtest",
    ],
)
