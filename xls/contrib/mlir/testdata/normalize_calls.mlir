// RUN: xls/contrib/mlir/xls_opt -normalize-xls-calls %s 2>&1 | FileCheck %s

// CHECK: xls.import{{.*}}"xls/contrib/mlir/testdata/dot_product.x"
// CHECK: func.func private{{.*}}fixed_test
// CHECK-LABEL: simple_call
func.func @simple_call(%arg0: !xls.array<4 x i32>, %arg1: !xls.array<4 x i32>) -> i32 {
  // CHECK: call @{{.*}}dot_product_fixed_test
  %0 = xls.call_dslx "xls/contrib/mlir/testdata/dot_product.x":
    "dot_product_fixed_test"(%arg0, %arg1) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  return %0 : i32
}
