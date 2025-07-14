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
# @rules_cc//cc:cc_binary.bzl

def libxls_dylib_binary(name = "libxls.dylib"):
    # Create a variant of the c_api_symbols.txt file that has leading
    # underscores on each symbol, as those are the OS X symbols.
    native.genrule(
        name = "c_api_symbols_underscores",
        srcs = [":c_api_symbols.txt"],
        outs = ["c_api_symbols_underscores.txt"],
        cmd = "sed 's/^/_/' < $(location :c_api_symbols.txt) > $@",
    )
    native.cc_binary(
        name = name,
        additional_linker_inputs = [
            ":exported_symbols_map",
            ":generate_linker_params_underscores",
            ":c_api_symbols_underscores.txt",
        ],
        copts = [
            "-fno-exceptions",
        ],
        linkopts = [
            "@$(location :generate_linker_params_underscores)",
            "-exported_symbols_list $(location :c_api_symbols_underscores.txt)",
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
            ":c_api_ir_builder",
            ":c_api_symbols.txt",
            ":c_api_vast",
        ],
    )
