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

"""Bzlmod extension for llvm-project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _llvm_raw_ext_impl(_ctx):
    LLVM_COMMIT = "0b1fe74f079fd330e4d4d85ce33b7f5702da1be7"
    LLVM_SHA256 = "849452cc51d3994aaff858e607d1148cd38b84ebae18f4b00b028c09879caec4"

    http_archive(
        name = "llvm-raw",
        build_file_content = "# empty",
        sha256 = LLVM_SHA256,
        patches = [
            Label("@//dependency_support/llvm:llvm.patch"),
            Label("@//dependency_support/llvm:zlib-header.patch"),
            Label("@//dependency_support/llvm:run_lit.patch"),
            # TODO: mikex - Remove this patch once the bug is fixed.
            Label("@//dependency_support/llvm:llvm_repo_metadata.patch"),
        ],
        patch_args = ["-p1"],
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    )

llvm_raw_ext = module_extension(implementation = _llvm_raw_ext_impl)
