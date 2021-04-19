# XLS (go/xls) libraries ported from Go math library.

load("//xls/build_rules:build_defs.bzl", "dslx_jit_wrapper", "dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
)

licenses(["notice"])

exports_files(["LICENSE"])

dslx_test(
    name = "fpsqrt_32",
    srcs = ["fpsqrt_32.x"],
    # 2021-03-22 Takes too long.
    prove_unopt_eq_opt = False,
)

dslx_jit_wrapper(
    name = "fpsqrt_32_jit_wrapper",
    dslx_name = "fpsqrt_32",
    deps = [":fpsqrt_32_opt_ir"],
)

cc_test(
    name = "fpsqrt_32_test",
    srcs = ["fpsqrt_32_test.cc"],
    data = [":fpsqrt_32_all_ir"],
    tags = ["optonly"],
    deps = [
        ":fpsqrt_32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)
