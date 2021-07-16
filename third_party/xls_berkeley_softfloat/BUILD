# XLS libraries ported from Berkeley SoftFloat.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_library",
    "xls_dslx_module_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
)

licenses(["notice"])

exports_files(["LICENSE"])

xls_dslx_library(
    name = "fpdiv_2x32_dslx",
    srcs = ["fpdiv_2x32.x"],
)

xls_dslx_module_library(
    name = "fpdiv_2x32_dslx_module",
    src = "fpdiv_2x32.x",
    deps = [":fpdiv_2x32_dslx"],
)

xls_dslx_opt_ir(
    name = "fpdiv_2x32",
    dep = ":fpdiv_2x32_dslx_module",
    ir_file = "fpdiv_2x32.opt.ir",
    tags = ["optonly"],
)

# TODO: Add xls_ir_equivalence_test, xls_eval_ir_test and xls_benchmark_ir.
# udiv not supported in delay model.
# udiv not supported in equivalence checking.
xls_dslx_test(
    name = "fpdiv_2x32_dslx_test",
    dep = ":fpdiv_2x32_dslx_module",
    tags = ["optonly"],
)

cc_xls_ir_jit_wrapper(
    name = "fpdiv_2x32_jit_wrapper",
    src = ":fpdiv_2x32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpdiv_2x32",
    },
)

cc_test(
    name = "fpdiv_2x32_test",
    srcs = ["fpdiv_2x32_test.cc"],
    tags = ["optonly"],
    deps = [
        ":fpdiv_2x32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench_builder",
    ],
)
