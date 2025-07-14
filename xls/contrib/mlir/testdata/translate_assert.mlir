// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=XLS

// XLS: token = assert
// XLS-SAME: message="Assertion failure"
// XLS-SAME: label="some_label"
func.func @test_arithop(%arg0: !xls.token, %arg1: i1, %arg2: i32) -> i32 { 
  %0 = xls.assert %arg0, %arg1, "Assertion failure", "some_label" 
  %1 = xls.not %arg2 : i32 
  return %1 : i32 
} 