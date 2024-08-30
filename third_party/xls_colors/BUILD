# XLS color library ported from https://www.vagrearg.org/content/hsvrgb

load("@rules_license//rules:license.bzl", "license")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_benchmark_ir",
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
    package_name = "xls_colors",
)

licenses(["notice"])

exports_files(["LICENSE"])

xls_dslx_test(
    name = "hsv2rgb_dslx_test",
    srcs = ["hsv2rgb.x"],
    dslx_test_args = {"compare": "jit"},
)

cc_test(
    name = "hsv2rgb_verilog_test",
    srcs = ["test.cc"],
    data = [
        "hsv2rgb.x",
    ],
    shard_count = 6,
    tags = ["optonly"],
    deps = [
        ":fast_hsv2rgb",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/ir_convert:convert_options",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/ir:bits",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "fast_hsv2rgb",
    srcs = ["fast_hsv2rgb_32bit.c"],
    hdrs = ["fast_hsv2rgb.h"],
)

xls_dslx_library(
    name = "hsv2rgb_dslx",
    srcs = ["hsv2rgb.x"],
    deps = [],
)

xls_dslx_opt_ir(
    name = "hsv2rgb",
    dslx_top = "hsv2rgb",
    ir_file = "hsv2rgb.ir",
    library = ":hsv2rgb_dslx",
    opt_ir_file = "hsv2rgb.opt.ir",
)

xls_benchmark_ir(
    name = "hsv2rgb_benchmark_ir",
    src = ":hsv2rgb.ir",
)
