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
        url = "https://github.com/nelhage/rules_boost/archive/1e3a69bf2d5cd10c34b74f066054cd335d033d71.tar.gz",
        strip_prefix = "rules_boost-1e3a69bf2d5cd10c34b74f066054cd335d033d71",
        sha256 = "b3cbdceaa95b8cfe3a69ff37f8ad0e53a77937433234f6b9a6add2eff5bde333",
        patches = [
            # rules_boost does not include Boost Python, see
            # https://github.com/nelhage/rules_boost/issues/67
            # This is because there doesn't exist a nice standard way in
            # Bazel to depend on the Python headers of the current Python
            # toolchain. The patch below selects the same Python headers
            # that the rest of XLS uses.
            "@com_google_xls//dependency_support/boost:add_python.patch",
        ],
    )
