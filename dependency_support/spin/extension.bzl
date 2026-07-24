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

"""Module extension for the SPIN model checker (https://spinroot.com)."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _spin_extension_impl(_ctx):
    http_archive(
        name = "spin",
        sha256 = "e2ba8cfebf963cb524be2a3caee68821472602f9c4f0e557f2550d8e382eab99",
        strip_prefix = "Spin-master",
        urls = ["https://github.com/nimble-code/Spin/archive/refs/heads/master.tar.gz"],
        build_file = Label("//dependency_support/spin:bundled.BUILD.bazel"),
        patches = [
            # Adds -Q <file>: emit one JSON line per channel event during simulation.
            # Adds -K<dir>: pan writes .trail files into <dir> instead of cwd.
            # Adds -H: pan exits with code 1 when errors > 0 (default is 0).
            Label("//dependency_support/spin/patches:spin.patch"),
        ],
        patch_args = ["-p1", "--fuzz=3"],
    )

spin_extension = module_extension(implementation = _spin_extension_impl)
