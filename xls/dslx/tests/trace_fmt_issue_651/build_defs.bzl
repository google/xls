# Copyright 2024 The XLS Authors
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

"""Convenience macro for tests that need to generate a JIT wrapper for a DSLX file.

Note: these are not user-facing, these are just for convenience of internal
development in specifying the various targets we may need as part of defining a DSLX
language-level test.
"""

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_ir",
)

def xls_dslx_ir_wrapper(name, library = ":test_target_library"):
    """
    Generate a JIT wrapper for a function in a DSLX file, for testing purposes.

    In the BUILD file you declare the wrapper target and then depend on it from
    the cc_test target:

        xls_dslx_ir_wrapper(
            name = "trace_u21_hex_wrapper",
        )

        cc_test(
            name = "trace_fmt_test",
            srcs = ["trace_fmt_test.cc"],
            deps = [
                ":trace_u21_hex_wrapper",
            ...

    Then, in a test file you import the generated wrapper and use it to run the DSLX function:

        #include "xls/dslx/tests/trace_fmt_issue_651/trace_[name]_wrapper.h"
        XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::[name]::Create());

    Args:
      name: The base name of the target, also the name of the `dslx_top` function.
      library: The `xls_dslx_library` rule that contains the DSLX file.
    """

    xls_dslx_ir(
        name = "%s_ir" % name,
        dslx_top = name,
        library = library,
    )

    cc_xls_ir_jit_wrapper(
        name = "%s_wrapper" % name,
        src = ":%s_ir.ir" % name,
        jit_wrapper_args = {
            "class_name": name.capitalize(),
            "namespace": "xls::wrapped",
        },
    )
