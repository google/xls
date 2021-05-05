# XLS (go/xls) libraries ported from Berkeley SoftFloat.

load("//xls/build_rules:build_defs.bzl", "dslx_jit_wrapper", "dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
)

licenses(["notice"])

exports_files(["LICENSE"])

dslx_test(
    name = "fpdiv_2x32",
    srcs = ["fpdiv_2x32.x"],
    # udiv not supported in delay model.
    generate_benchmark = False,
    # udiv not supported in equivalence checking.
    prove_unopt_eq_opt = False,
    tags = ["optonly"],
    deps = [
    ],
)

dslx_jit_wrapper(
    name = "fpdiv_2x32_jit_wrapper",
    dslx_name = "fpdiv_2x32",
    deps = [":fpdiv_2x32_opt_ir"],
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
        "//xls/tools:testbench",
    ],
)

exports_files(["fpdiv_2x32.x"])
