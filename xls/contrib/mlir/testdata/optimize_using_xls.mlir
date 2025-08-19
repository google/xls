// RUN: xls_opt %s -optimize-using-xls | FileCheck %s
// RUN: xls_opt %s -optimize-using-xls=xls-pipeline='""' | FileCheck %s --check-prefix=NOP

module @pkg {
xls.import_dslx_file_package "xls/contrib/mlir/testdata/i32/dot_product.x" as @dot_product

func.func private @bar2(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> i32 attributes
  {xls.linkage = #xls.translation_linkage<@dot_product:"dot_product_fixed_test">}

func.func private @bar(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> i32 {
  %1 = func.call @bar2(%a, %b) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  return %1 : i32
}

// Verifies all inlined post optimization roundtrip.
// CHECK: module @pkg {
// CHECK-NEXT: func @pkg
// CHECK-NOT: func
func.func @pkg(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> (i32) {
  %1 = func.call @bar(%a, %b) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  return %1 : i32
}

// Verifies that optimizations are skipped when the pipeline is set to <nop>.
// NOP: module @pkg {
// NOP: func.func
// NOP: func.func
// NOP: func.func
}
