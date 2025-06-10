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

"""Registers Bazel workspaces for the Boost C++ libraries."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "com_github_nelhage_rules_boost",
        url = "https://github.com/nelhage/rules_boost/archive/1217caae292dc9f14e8109777ba43c988cf89c5b.tar.gz",
        strip_prefix = "rules_boost-1217caae292dc9f14e8109777ba43c988cf89c5b",
        sha256 = "b3280ba677fb53c08dfe9fc3d77504c53efd31288c3b8445711c2e7b82281315",
        patches = [
            # rules_boost does not include Boost Python, see
            # https://github.com/nelhage/rules_boost/issues/67
            # This is because there doesn't exist a nice standard way in
            # Bazel to depend on the Python headers of the current Python
            # toolchain. The patch below selects the same Python headers
            # that the rest of XLS uses.
            Label("//dependency_support/boost:add_python.patch"),
            Label("//dependency_support/boost:backtrace_from_rule.patch"),
            # See: https://github.com/nelhage/rules_boost/issues/555
            Label("//dependency_support/boost:downgrade_lzma.patch"),
        ],
    )
