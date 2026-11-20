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

def get_src_files_from_dslx(library = None, srcs = None, deps = None):
    """Returns the DSLX source files from a DSLX library or srcs/deps lists.

    Args:
      library: A DSLX library target, or None.
      srcs: A list of DSLX source Files, or None.
      deps: A list of dependency targets, or None.

    Returns:
      Returns the DSLX source files.
    """
    dslx_src_files = []
    count = 0

    if library:
        dslx_info = library[DslxInfo]
        dslx_src_files = dslx_info.target_dslx_source_files
        count += 1
    if srcs or deps:
        if not srcs:
            fail("'srcs' must be defined when 'deps' is defined.")
        dslx_src_files = srcs
        count += 1

    if count != 1:
        fail("One of: 'library' or ['srcs', 'deps'] must be assigned.")

    return dslx_src_files

def get_src_files_from_dslx_library_as_input(ctx):
    """Returns the DSLX source files of rules using 'xls_dslx_library_as_input_attrs'.

    Args:
      ctx: The current rule's context object.

    Returns:
      Returns the DSLX source files of rules using 'xls_dslx_library_as_input_attrs'.
    """
    return get_src_files_from_dslx(
        library = ctx.attr.library,
        srcs = ctx.files.srcs,
        deps = ctx.attr.deps,
    )

def get_transitive_dslx_srcs_from_ctx(ctx, extra_srcs = []):
    """Returns all transitive DSLX source files reachable from ctx.

    Args:
      ctx: The current rule's context object.
      extra_srcs: Optional list of additional Files to include.

    Returns:
      A list of transitive DSLX source Files.
    """
    depsets = []
    if getattr(ctx.attr, "library", None) and DslxInfo in ctx.attr.library:
        depsets.append(ctx.attr.library[DslxInfo].dslx_source_files)
    if getattr(ctx.attr, "dep", None) and DslxInfo in ctx.attr.dep:
        depsets.append(ctx.attr.dep[DslxInfo].dslx_source_files)
    for dep in getattr(ctx.attr, "deps", []):
        if DslxInfo in dep:
            depsets.append(dep[DslxInfo].dslx_source_files)
    return depset(direct = list(extra_srcs), transitive = depsets).to_list()

def build_dslx_path(
        ctx,
        tool_label,
        dslx_srcs,
        user_dslx_path = "",
        is_test = False):
    """Builds a colon-separated --dslx_path value for DSLX tools and tests.

    Args:
      ctx: The current rule's context object (uses genfiles_dir and bin_dir).
      tool_label: Label of the XLS tool target being invoked (used to locate
        the XLS repository root for the DSLX standard library when XLS is an
        external repository).
      dslx_srcs: Sequence of DSLX source Files (direct and transitive) whose
        workspaces and output roots should be searchable.
      user_dslx_path: Optional user-provided colon-separated search path prefix.
      is_test: True when building a command line for a test shell wrapper
        (runs in $TEST_SRCDIR/<main_repo>, where external repos live at
        ../<workspace_name>); False for build actions (runs in execroot, where
        external repos live at external/<workspace_name>).

    Returns:
      A colon-separated search path string suitable for --dslx_path.
    """
    entries = []
    seen = {}

    def _add(path):
        if path and path not in seen:
            seen[path] = True
            entries.append(path)

    if user_dslx_path:
        for part in user_dslx_path.split(":"):
            _add(part)

    _add("${PWD}" if is_test else ".")
    _add(ctx.genfiles_dir.path)
    _add(ctx.bin_dir.path)

    for src in dslx_srcs:
        if src.root.path:
            _add(src.root.path)
        owner = src.owner
        if owner:
            if is_test and owner.workspace_name:
                _add("../" + owner.workspace_name)
            if owner.workspace_root:
                _add(owner.workspace_root)
                if src.root.path:
                    _add(src.root.path + "/" + owner.workspace_root)
                _add(ctx.genfiles_dir.path + "/" + owner.workspace_root)
                _add(ctx.bin_dir.path + "/" + owner.workspace_root)

    if tool_label:
        if is_test and tool_label.workspace_name:
            _add("../" + tool_label.workspace_name)
        if tool_label.workspace_root:
            _add(tool_label.workspace_root)

    return ":".join(entries)
