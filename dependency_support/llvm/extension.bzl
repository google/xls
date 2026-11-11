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
    LLVM_COMMIT = "bb04dac04a0f9e9b99ffc2e28b4806d4f98e7966"
    LLVM_SHA256 = "8e70f7285da2084fca6365c65e0ab520f8c105cef58404c7634fb1e7425c37be"

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
