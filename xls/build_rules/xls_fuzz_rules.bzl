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

"""This module contains the dslx_fuzz_test macro."""

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//xls/build_rules:xls_ir_macros.bzl",
    xls_dslx_ir = "xls_dslx_ir_macro",
)
load(
    "//xls/build_rules:xls_jit_wrapper_rules.bzl",
    "FUNCTION_WRAPPER_TYPE",
    "FUZZTEST_WRAPPER_TYPE",
    "cc_xls_ir_jit_wrapper",
)
load(
    "//xls/build_rules:xls_macros.bzl",
    xls_dslx_opt_ir = "xls_dslx_opt_ir_macro",
)

def dslx_fuzz_test(
        name,
        library,
        test_function,
        opt_ir = False,
        jit_namespace = "dslx_fuzztest::impl",
        tags = []):
    """A macro that instantiates a DSLX fuzz test.

    The macro instantiates:
    1. An xls_dslx_ir target to generate non-optimized IR.
    2. A cc_xls_ir_jit_wrapper target to generate C++ fuzz test wrapper code.
    3. A cc_test target to run the fuzz test.

    Args:
      name: The name of the test target.
      library: The xls_dslx_library target containing the DSLX code.
      opt_ir: Whether to generate optimized IR or not.
      test_function: The name of the top-level DSLX function to be fuzzed.
      jit_namespace: The C++ namespace for the generated JIT wrapper class.
      tags: Tags to apply to the generated cc_test target.
    """
    if opt_ir:
        ir_name = name + "_opt_ir"
        ir_file = name + "_" + test_function + ".opt.ir"
        build_ir = xls_dslx_opt_ir
    else:
        ir_name = name + "_ir"
        ir_file = name + "_" + test_function + ".ir"
        build_ir = xls_dslx_ir

    build_ir(
        name = ir_name,
        library = library,
        dslx_top = test_function,
        ir_file = ir_file,
    )

    lib_name = name + "_lib"
    lib_class_name = name + "_" + test_function + "_lib"
    cc_xls_ir_jit_wrapper(
        name = lib_name,
        src = ":" + ir_name,
        wrapper_type = FUNCTION_WRAPPER_TYPE,
        jit_wrapper_args = {
            "class_name": lib_class_name,
            "namespace": jit_namespace,
        },
        enable_llvm_coverage = True,
        testonly = True,
    )

    fuzzer_name = name + "_fuzzer"
    lib_header_path = "%s/%s.h" % (native.package_name(), lib_name)
    cc_xls_ir_jit_wrapper(
        name = fuzzer_name,
        src = ":" + ir_name,
        wrapper_type = FUZZTEST_WRAPPER_TYPE,
        jit_wrapper_args = {
            "class_name": name + "_" + test_function + "_Fuzzer",
            "namespace": jit_namespace,
            "lib_class_name": jit_namespace + "::" + lib_class_name,
            "lib_header_path": lib_header_path,
        },
        testonly = True,
        deps = [":" + lib_name],
    )

    cc_test(
        name = name,
        srcs = [],
        deps = [
            ":" + fuzzer_name,
            "@googletest//:gtest",
            "//xls/common/fuzzing:fuzztest",
            "//xls/common:xls_gunit_main",
            "//xls/ir:value",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "//xls/common/status:status_macros",
            "//xls/jit:function_base_jit_wrapper",
            "//xls/ir:bits",
            "//xls/ir",
            "//xls/ir:type",
            "//xls/ir:value_test_util",
        ],
        tags = tags,
        testonly = True,
    )
