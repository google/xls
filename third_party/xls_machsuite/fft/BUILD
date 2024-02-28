load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//third_party/xls_machsuite:license"],
    licenses = ["notice"],
)

cc_binary(
    name = "fft_reference",
    srcs = ["fft_reference.cc"],
    data = [
        "test_data/img_in",
        "test_data/img_twid_in",
        "test_data/real_in",
        "test_data/real_twid_in",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
    ],
)

xls_dslx_library(
    name = "fft_dslx",
    srcs = [
        "fft.x",
        "test_data/dslx_test_data.x",
    ],
    tags = ["optonly"],
)

# IR optimization uses too much memory.
# TODO(https://github.com/google/xls/issues/354): 2021-03-23 Fix this. Add
# xls_ir_equivalence_test, xls_eval_ir_test and xls_benchmark_ir.
xls_dslx_test(
    name = "fft_test",
    size = "large",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":fft_dslx",
    tags = ["optonly"],
)
