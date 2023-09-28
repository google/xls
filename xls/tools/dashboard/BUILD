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

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "mkdocs_creator",
    srcs = ["mkdocs_creator.py"],
    deps = [
        requirement("mkdocs"),
        requirement("mkdocs-material"),
    ],
)

py_library(
    name = "json_to_markdown",
    srcs = ["json_to_markdown.py"],
    deps = [requirement("mdutils")],
)

py_library(
    name = "validate_dashboard_json",
    srcs = ["validate_dashboard_json.py"],
    deps = [requirement("jsonschema")],
)

py_library(
    name = "run_and_parse",
    srcs = ["run_and_parse.py"],
    deps = [
        ":validate_dashboard_json",
    ],
)

py_binary(
    name = "dashboard",
    srcs = ["dashboard.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":json_to_markdown",
        ":mkdocs_creator",
        ":run_and_parse",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":validate_dashboard_json",
    ],
)

py_binary(
    name = "cocotb_results_xml_parser",
    srcs = ["cocotb_results_xml_parser.py"],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_binary(
    name = "dslx_test_parser",
    srcs = ["dslx_test_parser.py"],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_binary(
    name = "generic_parser",
    srcs = ["generic_parser.py"],
    python_version = "PY3",
    srcs_version = "PY3",
)
