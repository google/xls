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

"""
This module contains build macros that are specific to XLS's OSS release.

This module is intended to be loaded by the xls/public/BUILD file.
"""

# pytype tests are present in this file

def libxls_dylib_binary(name = "libxls.dylib"):
    native.cc_binary(
        name = name,
        additional_linker_inputs = [
            ":exported_symbols_map",
            ":generate_linker_params_underscores",
        ],
        copts = [
            "-fno-exceptions",
        ],
        linkopts = [
            "@$(location :generate_linker_params_underscores)",
        ],
        linkshared = True,
        target_compatible_with = select({
            "@platforms//os:osx": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        deps = [
            ":c_api",
        ],
    )

def pytype_test_test_c_api_symbols(name = "test_c_api_symbols"):
    native.py_test(
        name = name,
        srcs = ["test_c_api_symbols.py"],
        data = [
            ":c_api",
            ":c_api_dslx",
            ":c_api_symbols.txt",
            ":c_api_vast",
        ],
    )
