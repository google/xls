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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "simple_vc_experiment",
    srcs = ["simple_vc_experiment.cc"],
    hdrs = ["simple_vc_experiment.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "//xls/noc/drivers:experiment",
        "//xls/noc/drivers:experiment_factory",
        "//xls/noc/simulation:common",
        "//xls/noc/simulation:traffic_description",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "aggregate_tree_experiment",
    srcs = ["aggregate_tree_experiment.cc"],
    hdrs = ["aggregate_tree_experiment.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/noc/config:network_config_cc_proto",
        "//xls/noc/config:network_config_proto_builder",
        "//xls/noc/drivers:experiment",
        "//xls/noc/drivers:experiment_factory",
        "//xls/noc/simulation:common",
        "//xls/noc/simulation:traffic_description",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)
