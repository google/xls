# XLS (go/xls) libraries ported from Go math library.

load("//xls/build_rules:build_defs.bzl", "dslx_jit_wrapper", "dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
)

licenses(["notice"])

exports_files(["LICENSE"])

dslx_test(
    name = "fpexp_32",
    srcs = ["fpexp_32.x"],
    # 2021-03-08: Takes too long (> 15 minutes).
    prove_unopt_eq_opt = False,
    deps = [
        "//xls/modules:apfloat_add_2.x",
        "//xls/modules:apfloat_mul_2.x",
        "//xls/modules:apfloat_sub_2.x",
        "//xls/modules:fpadd_2x32.x",
        "//xls/modules:fpldexp_32.x",
        "//xls/modules:fpmul_2x32.x",
        "//xls/modules:fpsub_2x32.x",
        "//third_party/xls_berkeley_softfloat:fpdiv_2x32.x",
    ],
)

dslx_jit_wrapper(
    name = "fpexp_32_jit_wrapper",
    dslx_name = "fpexp_32",
    deps = [":fpexp_32_opt_ir"],
)

cc_test(
    name = "fpexp_32_test",
    srcs = ["fpexp_32_test.cc"],
    data = [":fpexp_32_all_ir"],
    tags = ["optonly"],
    deps = [
        ":fpexp_32_jit_wrapper",
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

dslx_test(
    name = "fp_sincos_32",
    srcs = ["fp_sincos_32.x"],
    # TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes).
    prove_unopt_eq_opt = False,
    deps = [
        "fp_trig_reduce.x",
        "//xls/modules:apfloat_add_2.x",
        "//xls/modules:apfloat_mul_2.x",
        "//xls/modules:apfloat_sub_2.x",
        "//xls/modules:fpadd_2x32.x",
        "//xls/modules:fpmul_2x32.x",
        "//xls/modules:fpsub_2x32.x",
    ],
)

dslx_jit_wrapper(
    name = "fp_sincos_32_jit_wrapper",
    dslx_name = "fp_sincos_32",
    deps = [":fp_sincos_32_opt_ir"],
)

cc_test(
    name = "fp_sincos_32_test",
    srcs = ["fp_sincos_32_test.cc"],
    data = [":fp_sincos_32_all_ir"],
    tags = ["optonly"],
    deps = [
        ":fp_sincos_32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

dslx_test(
    name = "fp_trig_reduce",
    srcs = ["fp_trig_reduce.x"],
    # TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes).
    prove_unopt_eq_opt = False,
    deps = [
        "//xls/modules:apfloat_add_2.x",
        "//xls/modules:apfloat_mul_2.x",
        "//xls/modules:apfloat_sub_2.x",
        "//xls/modules:fpadd_2x64.x",
        "//xls/modules:fpmul_2x64.x",
        "//xls/modules:fpsub_2x64.x",
    ],
)

dslx_jit_wrapper(
    name = "fp_trig_reduce_jit_wrapper",
    dslx_name = "fp_trig_reduce",
    deps = [":fp_trig_reduce_opt_ir"],
)

cc_test(
    name = "fp_trig_reduce_64_test",
    srcs = ["fp_trig_reduce_64_test.cc"],
    data = [":fp_trig_reduce_all_ir"],
    tags = ["optonly"],
    deps = [
        ":fp_trig_reduce_jit_wrapper",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

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
