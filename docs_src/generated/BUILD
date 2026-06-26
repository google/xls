# Copyright 2026 The XLS Authors
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

load("@stardoc//stardoc:stardoc.bzl", "stardoc")
load("//xls/build_rules:xls_build_defs.bzl", "xls_diff_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

stardoc(
    name = "xls_stardoc",
    out = "xls_bazel_rules_macros_generated.md",
    format = "markdown",
    header_template = "stardoc_header.vm",
    input = "//xls/build_rules:xls_build_defs.bzl",
    # TODO(allight): We probably want this to be true but that makes migration
    # to/from OSS more complex.
    render_main_repo_name = False,
    deps = ["//xls/build_rules:xls_build_defs_bzl"],
)

# Test to keep the XLS build rules/macros document in sync.
xls_diff_test(
    name = "xls_rules_stardoc_diff_test",
    failure_message = """Generated file :xls_bazel_rules_macros_generated.md differs from checked in bazel_rules_macros.md.

To regenerate, run: `./xls/dev_tools/rebuild_golden_files.sh`""",
    file = ":xls_bazel_rules_macros_generated.md",
    golden = "//docs_src:bazel_rules_macros.md",
)

stardoc(
    name = "xlscc_stardoc",
    out = "xlscc_bazel_rules_macros_generated.md",
    format = "markdown",
    header_template = "stardoc_header.vm",
    input = "//xls/contrib/xlscc/build_rules:xlscc_build_defs.bzl",
    # TODO(allight): We probably want this to be true but that makes migration
    # to/from OSS more complex.
    render_main_repo_name = False,
    deps = ["//xls/contrib/xlscc/build_rules:xlscc_build_defs_bzl"],
)

# Test to keep the XLScc build rules/macros document in sync.
xls_diff_test(
    name = "xlscc_rules_stardoc_diff_test",
    failure_message = """Generated file :xlscc_bazel_rules_macros_generated.md differs from checked in xls/contrib/xlscc/bazel_rules_macros.md.

To regenerate, run: `./xls/dev_tools/rebuild_golden_files.sh`""",
    file = ":xlscc_bazel_rules_macros_generated.md",
    golden = "//xls/contrib/xlscc:bazel_rules_macros.md",
)
