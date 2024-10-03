// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-xls %s --main-function=sub -- | FileCheck %s --check-prefix=XLS
// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-xls --optimize-ir %s --main-function=sub -- 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=XLS-OPT

// XLS: fn sub
// XLS-DAG: invoke{{.*}}=__struct_type__int_to_float
// XLS-DAG: invoke{{.*}}=__dot_product__dot_product_fixed_test
// XLS-DAG: invoke{{.*}}=__struct_type__int_to_float
// XLS-DAG: invoke{{.*}}=__struct_type__float_to_int
// XLS-DAG: invoke{{.*}}=__dot_product__dot_product_fixed_test

// XLS-OPT-NOT: fn {{.*}}dot_product
// XLS-OPT: fn sub

module @pkg {
xls.import_dslx_file_package "xls/contrib/mlir/testdata/dot_product.x" as @dot_product
xls.import_dslx_file_package "xls/contrib/mlir/testdata/struct_type.x" as @struct_type

// This maps a static function to static one.
func.func private @bar2(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> i32 attributes
  {xls.linkage = #xls.translation_linkage<@dot_product:"dot_product_fixed_test">}

// This maps a static function to static one with an opaque return type.
func.func private @int_to_float_opaque(%a: i32) -> !xls.opaque<"a"> attributes
  {xls.linkage = #xls.translation_linkage<@struct_type:"int_to_float">}

// This maps a static function to static one with an opaque parameter type.
func.func private @float_to_int_opaque(%a: !xls.opaque<"a">) -> i32 attributes
  {xls.linkage = #xls.translation_linkage<@struct_type:"float_to_int">}

// This imports + instantiates function.
func.func private @bar(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> i32 attributes
  {xls.linkage = #xls.translation_linkage<@dot_product:
    "fn dot_product_fixed_test(a : s32[4], b: s32[4]) -> s32 { dot_product_fixed<u32:32, u32:4>(a, b) }"> }

func.func private @int_to_float32(%a: i32) -> tuple<ui1, ui8, ui23> attributes
  {xls.linkage = #xls.translation_linkage<@struct_type:"int_to_float">}

func.func @sub(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>, %c: i32, %d: !xls.opaque<"a">) -> (i32, !xls.opaque<"a">) {
  %0 = func.call @bar(%a, %b) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  %1 = func.call @bar2(%a, %b) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  %2 = func.call @int_to_float32(%c) : (i32) -> tuple<ui1, ui8, ui23>
  %3 = func.call @int_to_float_opaque(%c) : (i32) -> !xls.opaque<"a">
  %4 = func.call @float_to_int_opaque(%3) : (!xls.opaque<"a">) -> i32
  return %0, %d : i32, !xls.opaque<"a">
}

xls.chan @mychan : i32

xls.eproc @eproc(%arg: i32) zeroinitializer {
  %tok = xls.after_all : !xls.token
  %tok2 = xls.send %tok, %arg, @mychan : i32
  %tok3, %result = xls.blocking_receive %tok2, @mychan : i32
  xls.yield %arg : i32
}

xls.import_dslx_file_package "xls/contrib/mlir/stdlib/fp_ext_trunc.x" as @ext_trunclib
func.func private @ext_trunclib_trunc(f32) -> bf16 attributes {xls.linkage = #xls.translation_linkage<@ext_trunclib : "trunc">}
func.func private @ext_trunclib_ext(bf16) -> f32 attributes {xls.linkage = #xls.translation_linkage<@ext_trunclib : "ext">}

func.func private @trunc_ext(%arg0: f32) -> f32 attributes {xls = true} {
  %0 = func.call @ext_trunclib_trunc(%arg0) : (f32) -> bf16
  %1 = func.call @ext_trunclib_ext(%0) : (bf16) -> f32
  return %1 : f32
}

}

