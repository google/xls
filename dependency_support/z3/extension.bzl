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

"""Module extension for Z3."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _z3_extension_impl(
        module_ctx):  # @unused
    http_archive(
        name = "z3",
        urls = ["https://github.com/Z3Prover/z3/archive/z3-4.14.1.tar.gz"],
        integrity = "sha256-gaAsLGTGTWw98jP1kYa5VieZCtoMTC/JAcnCWnByZyo=",
        strip_prefix = "z3-z3-4.14.1",
        build_file = Label("//dependency_support/z3:bundled.BUILD.bazel"),
    )

z3_extension = module_extension(
    implementation = _z3_extension_impl,
)
