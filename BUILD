# Copyright 2022 The XLS Authors
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

# NOTE: this is the build file for the root directory of the workspace. The
# root directory generally has very few files of interest.
#
# Targets for XLS are generally in the `xls` subdirectory; e.g.
# can be seen via:
#
#   bazel query //xls/...
#
# And built (in optimized mode) via:
#
#   bazel build -c opt //xls/...
#
# See https://google.github.io/xls/build_system/#whirlwind-intro-to-bazel for
# more tutorial information.

load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = ["//xls:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

license(
    name = "license",
    package_name = "xls",
)

exports_files(["LICENSE", "mkdocs.yml"])
