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

"""This module contains common functions for DSLX build rules for XLS."""

load(
    "//xls/build_rules:xls_providers.bzl",
    "DslxInfo",
)

visibility(["//xls/build_rules/..."])

xls_dslx_library_as_input_attrs = {
    "library": attr.label(
        doc = "A DSLX library target where the direct (non-transitive) " +
              "files of the target are tested. This attribute is mutually " +
              "exclusive with the 'srcs' and 'deps' attribute.",
        providers = [DslxInfo],
    ),
    "srcs": attr.label_list(
        doc = "Source files for the rule. The files must have a '.x' " +
              "extension. This attribute is mutually exclusive with the " +
              "'library' attribute.",
        allow_files = [".x"],
    ),
    "deps": attr.label_list(
        doc = "Dependency targets for the files in the 'srcs' attribute. " +
              "This attribute is mutually exclusive with the 'library' " +
              "attribute.",
        providers = [DslxInfo],
    ),
}

def get_src_files_from_dslx_library_as_input(ctx):
    """Returns the DSLX source files of rules using 'xls_dslx_library_as_input_attrs'.

    Args:
      ctx: The current rule's context object.

    Returns:
      Returns the DSLX source files of rules using 'xls_dslx_library_as_input_attrs'.
    """
    dslx_src_files = []
    count = 0

    if ctx.attr.library:
        dslx_info = ctx.attr.library[DslxInfo]
        dslx_src_files = dslx_info.target_dslx_source_files
        count += 1
    if ctx.attr.srcs or ctx.attr.deps:
        if not ctx.attr.srcs:
            fail("'srcs' must be defined when 'deps' is defined.")
        dslx_src_files = ctx.files.srcs
        count += 1

    if count != 1:
        fail("One of: 'library' or ['srcs', 'deps'] must be assigned.")

    return dslx_src_files
