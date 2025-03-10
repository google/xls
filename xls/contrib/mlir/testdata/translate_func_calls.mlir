// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

// CHECK: fn bar
// CHECK-NEXT: ret tuple
func.func private @bar(%arg0: i32, %arg1: i8) -> (i32, i8) {
  func.return %arg0, %arg1 : i32, i8
}

// CHECK: fn foo
// CHECK-NEXT: invoke
// CHECK: ret tuple_index
func.func @foo(%arg0: i32, %arg1: i8) -> i32 {
  %0:2 = func.call @bar(%arg0, %arg1) : (i32, i8) -> (i32, i8)
  func.return %0#0 : i32
}
