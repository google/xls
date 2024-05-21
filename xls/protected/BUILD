# Copyright 2022 The XLS Authors
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

# Protected APIs that we expect to change frequently and cannot be depended
# upon as they are internal XLS details.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_best_effort_users"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "call_graph",
    hdrs = ["call_graph.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "//xls/ir",
    ],
)

cc_library(
    name = "ir",
    hdrs = ["ir.h"],
    deps = [
        "//xls/ir",
    ],
)

cc_library(
    name = "netlist",
    hdrs = ["netlist.h"],
    deps = [
        "//xls/netlist",
        "//xls/netlist:cell_library",
        "//xls/netlist:function_extractor",
        "//xls/netlist:interpreter",
        "//xls/netlist:lib_parser",
        "//xls/netlist:netlist_parser",
    ],
)
