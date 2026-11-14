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
    LLVM_COMMIT = "de574ff51d4fff318028c235f20d3eda52e03915"
    LLVM_SHA256 = "f915797def09cb8495cbe3c6995432065096cf4a8b8fc4e5b1aee3b574b3c152"

    http_archive(
        name = "llvm-raw",
        build_file_content = "# empty",
        sha256 = LLVM_SHA256,
        patches = [
            Label("@//dependency_support/llvm:llvm.patch"),
            Label("@//dependency_support/llvm:run_lit.patch"),
        ],
        patch_args = ["-p1"],
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    )

llvm_raw_ext = module_extension(implementation = _llvm_raw_ext_impl)
