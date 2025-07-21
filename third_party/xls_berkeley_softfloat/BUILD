# XLS libraries ported from Berkeley SoftFloat.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_license//rules:license.bzl", "license")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//xls:xls_internal"],
)

license(
    name = "license",
    package_name = "xls_berkeley_softfloat",
)

licenses(["notice"])

exports_files(["LICENSE"])

xls_dslx_library(
    name = "fpdiv_2x32_dslx",
    srcs = ["fpdiv_2x32.x"],
)

xls_dslx_opt_ir(
    name = "fpdiv_2x32_opt_ir",
    dslx_top = "fpdiv_2x32",
    ir_file = "fpdiv_2x32.opt.ir",
    library = ":fpdiv_2x32_dslx",
    tags = ["optonly"],
)

# TODO: Add xls_ir_equivalence_test, xls_eval_ir_test and xls_benchmark_ir.
# udiv not supported in delay model.
# udiv not supported in equivalence checking.
xls_dslx_test(
    name = "fpdiv_2x32_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":fpdiv_2x32_dslx",
    tags = ["optonly"],
)

cc_xls_ir_jit_wrapper(
    name = "fpdiv_2x32_jit_wrapper",
    src = ":fpdiv_2x32.opt.ir",
    jit_wrapper_args = {
        "class_name": "Fpdiv2x32",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fpdiv_2x32_test",
    srcs = ["fpdiv_2x32_test.cc"],
    tags = ["optonly"],
    deps = [
        ":fpdiv_2x32_jit_wrapper",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)
