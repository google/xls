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

"""Provides a rule that creates a C++ library from an .so file."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

def _impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(ctx = ctx, cc_toolchain = cc_toolchain)
    linker_inputs = [cc_common.create_linker_input(
        owner = ctx.label,
        user_link_flags = depset([ctx.file.src.path]),
        additional_inputs = depset([ctx.file.src]),
    )]

    return [
        DefaultInfo(
            files = depset([ctx.file.src]),
        ),
        CcInfo(
            linking_context = cc_common.create_linking_context(
                linker_inputs = depset(linker_inputs),
            ),
        ),
    ]

cc_so_library = rule(
    implementation = _impl,
    fragments = ["cpp"],
    attrs = {
        "src": attr.label(allow_single_file = True),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    incompatible_use_toolchain_transition = True,
    toolchains = use_cpp_toolchain(),
)
