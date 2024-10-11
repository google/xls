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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//xls:xls_mlir_users",
    ],
    features = [
        "layering_check",
        "parse_headers",
    ],
)

td_library(
    name = "xls_ops_td_files",
    srcs = glob(["IR/*.td"]),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:CopyOpInterfaceTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:MathOpsTdFiles",
        "@llvm-project//mlir:MemRefOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "xls_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "IR/xls_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "IR/xls_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/xls_ops.td",
    deps = [":xls_ops_td_files"],
)

gentbl_cc_library(
    name = "xls_ops_attrs_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "IR/xls_ops_attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "IR/xls_ops_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/xls_ops.td",
    deps = [":xls_ops_td_files"],
)

gentbl_cc_library(
    name = "xls_transforms_passes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=XlsTransforms",
            ],
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "xls_ops_enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "IR/xls_ops_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "IR/xls_ops_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/xls_ops.td",
    deps = [":xls_ops_td_files"],
)

gentbl_cc_library(
    name = "interfaces_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "IR/interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "IR/interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/interfaces.td",
    deps = [":xls_ops_td_files"],
)

gentbl_cc_library(
    name = "xls_ops_typedefs_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=xls",
            ],
            "IR/xls_ops_typedefs.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=xls",
            ],
            "IR/xls_ops_typedefs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/xls_ops.td",
    deps = [":xls_ops_td_files"],
)

gentbl_cc_library(
    name = "arith_to_xls_patterns_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/arith_to_xls_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/arith_to_xls_patterns.td",
    deps = [":xls_ops_td_files"],
)

gentbl_cc_library(
    name = "math_to_xls_patterns_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/math_to_xls_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/math_to_xls_patterns.td",
    deps = [":xls_ops_td_files"],
)

cc_library(
    name = "assembly_format",
    srcs = ["IR/assembly_format.cc"],
    hdrs = ["IR/assembly_format.h"],
    deps = [
        ":xls_ops_typedefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_xls",
    srcs = [
        "IR/interfaces.cc.inc",
        "IR/interfaces.h.inc",
        "IR/xls_ops.cc",
        "IR/xls_ops.cc.inc",
        "IR/xls_ops.h.inc",
        "IR/xls_ops_attrs.cc.inc",
        "IR/xls_ops_attrs.h.inc",
        "IR/xls_ops_enums.cc.inc",
        "IR/xls_ops_enums.h.inc",
        "IR/xls_ops_typedefs.cc.inc",
        "IR/xls_ops_typedefs.h.inc",
    ],
    hdrs = [
        "IR/xls_ops.h",
    ],
    deps = [
        ":assembly_format",
        ":interfaces_inc_gen",
        ":xls_ops_attrs_inc_gen",
        ":xls_ops_enums_inc_gen",
        ":xls_ops_inc_gen",
        ":xls_ops_typedefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "index_type_conversion",
    srcs = ["transforms/index_type_conversion.cc"],
    deps = [
        ":mlir_xls",
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "arith_to_xls",
    srcs = ["transforms/arith_to_xls.cc"],
    deps = [
        ":arith_to_xls_patterns_inc_gen",
        ":mlir_xls",
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "xls_transforms_passes",
    hdrs = ["transforms/passes.h"],
    deps = [
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//mlir:IR",  # buildcleaner: keep
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "math_to_xls",
    srcs = ["transforms/math_to_xls.cc"],
    hdrs = ["transforms/math_to_xls.h"],
    deps = [
        ":math_to_xls_patterns_inc_gen",
        ":mlir_xls",
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "normalize_xls_calls",
    srcs = ["transforms/normalize_xls_calls.cc"],
    deps = [
        ":mlir_xls",
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "scf_to_xls",
    srcs = ["transforms/scf_to_xls.cc"],
    deps = [
        ":mlir_xls",
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "scalarize",
    srcs = ["transforms/scalarize.cc"],
    deps = [
        ":mlir_xls",
        ":xls_transforms_passes_inc_gen",
        "@com_google_absl//absl/algorithm:container",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "xls_dialect_registration",
    srcs = ["IR/init.cc"],
    hdrs = ["IR/register.h"],
    deps = [":mlir_xls"],
)

cc_library(
    name = "transforms/xls_lower",
    srcs = ["transforms/xls_lower.cc"],
    hdrs = ["transforms/xls_lower.h"],
    deps = [
        ":arith_to_xls",  # buildcleaner: keep
        ":expand_macro_ops",  # buildcleaner: keep
        ":index_type_conversion",  # buildcleaner: keep
        ":math_to_xls",  # buildcleaner: keep
        ":normalize_xls_calls",  # buildcleaner: keep
        ":proc_elaboration",  # buildcleaner: keep
        ":scalarize",  # buildcleaner: keep
        ":scf_to_xls",  # buildcleaner: keep
        ":transforms/lower_counted_for",  # buildcleaner: keep
        ":xls_transforms_passes",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "transforms/lower_counted_for",
    srcs = ["transforms/lower_counted_for.cc"],
    deps = [
        ":mlir_xls",
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "proc_elaboration",
    srcs = ["transforms/proc_elaboration.cc"],
    deps = [
        ":interpreter",
        ":mlir_xls",
        ":xls_transforms_passes",
        ":xls_transforms_passes_inc_gen",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "expand_macro_ops",
    srcs = ["transforms/expand_macro_ops.cc"],
    deps = [
        ":mlir_xls",
        ":xls_transforms_passes",
        ":xls_transforms_passes_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_binary(
    name = "xls_opt",
    srcs = ["tools/xls_opt/xls_opt.cc"],
    deps = [
        ":arith_to_xls",
        ":expand_macro_ops",
        ":extraction_utils",
        ":mlir_xls",
        ":normalize_xls_calls",
        ":proc_elaboration",
        ":scalarize",
        ":scf_to_xls",
        ":transforms/lower_counted_for",
        ":transforms/xls_lower",
        ":xls_dialect_registration",
        ":xls_transforms_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "xls_translate_lib",
    srcs = ["tools/xls_translate/xls_translate.cc"],
    hdrs = ["tools/xls_translate/xls_translate.h"],
    deps = [
        ":mlir_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:source_location",
        "//xls/public:function_builder",
        "//xls/public:ir",
        "//xls/public:ir_parser",
        "//xls/public:runtime_build_actions",
        "//xls/tools:codegen_flags",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:opt",
        "//xls/tools:scheduling_options_flags",
        "//xls/tools:scheduling_options_flags_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_binary(
    name = "xls_translate",
    srcs = ["tools/xls_translate/xls_translate_main.cc"],
    data = [
        "stdlib/fp_ext_trunc.x",
    ],
    deps = [
        ":xls_dialect_registration",
        ":xls_translate_lib",
        "//xls/public:c_api",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "interpreter",
    srcs = [],
    hdrs = ["util/interpreter.h"],
    deps = [
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "interpreter_test",
    srcs = ["util/interpreter_test.cc"],
    deps = [
        ":interpreter",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "extraction_utils",
    srcs = ["util/extraction_utils.cc"],
    hdrs = ["util/extraction_utils.h"],
    deps = [
        ":mlir_xls",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
