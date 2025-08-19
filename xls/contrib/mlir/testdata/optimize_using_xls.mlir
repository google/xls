// RUN: xls_opt %s -optimize-using-xls | FileCheck %s

module @pkg {
xls.import_dslx_file_package "xls/contrib/mlir/testdata/i32/dot_product.x" as @dot_product

func.func private @bar2(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> i32 attributes
  {xls.linkage = #xls.translation_linkage<@dot_product:"dot_product_fixed_test">}

// CHECK: module @pkg {
// CHECK-NOT: import
// CHECK-LABEL: func @pkg
// CHECK: xls.smul
func.func @pkg(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> (i32) {
  %1 = func.call @bar2(%a, %b) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  return %1 : i32
}

}