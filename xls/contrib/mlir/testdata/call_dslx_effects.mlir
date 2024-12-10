// RUN: xls_opt -canonicalize %s 2>&1 | FileCheck %s

// CHECK-LABEL: simple_call
func.func @simple_call(%arg0: !xls.array<4 x i32>, %arg1: !xls.array<4 x i32>) -> i32 {
  // CHECK: "f"
  // CHECK-NOT: "g"
  // CHECK: "h"
  %0 = xls.call_dslx "foo.x": "f"(%arg0, %arg1) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  %1 = xls.call_dslx "foo.x": "g"(%arg0, %arg1) {is_pure} : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  %2 = xls.call_dslx "foo.x": "h"(%arg0, %arg1) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  return %0 : i32
}
