# XLS color library ported from https://www.vagrearg.org/content/hsvrgb

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_module_library",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
)

licenses(["notice"])

exports_files(["LICENSE"])

xls_dslx_module_library(
    name = "hsv2rgb_dslx_module",
    src = "hsv2rgb.x",
)

xls_dslx_test(
    name = "hsv2rgb_dslx_test",
    dep = ":hsv2rgb_dslx_module",
)
