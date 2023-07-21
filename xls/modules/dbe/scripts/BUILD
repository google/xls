# Copyright 2023 The XLS Authors
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

# Build rules for LZ4 Python reference & tests

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],
)

load("@xls_pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "dbe",
    imports = [
        ".",
    ],
    srcs = [
        "dbe/__init__.py",
        "dbe/common.py",
        "dbe/decoder.py",
        "dbe/random.py",
        "dbe/utils.py",
    ],
)

py_binary(
    name = "randgen_tok",
    srcs = [
        "randgen_tok.py"
    ],
    deps = [
        ":dbe",
    ],
)
