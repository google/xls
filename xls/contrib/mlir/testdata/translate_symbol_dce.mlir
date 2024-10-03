// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-xls --main-function=combinational --privatize-and-dce-functions %s -- 2>&1 | FileCheck %s

// CHECK-NOT: unused_function
module @pkg {
func.func @unused_function(%a: i8, %b: i8, %c: i8) -> i8 {
  %diff = xls.sub %a, %b: i8
  %umul.5 = xls.umul %diff, %diff : i8
  %umul.6 = xls.umul %c, %diff : i8
  %the_output = xls.add %umul.5, %umul.6 : i8
  return %the_output : i8
}

func.func @combinational(%a: i8, %b: i8, %c: i8) -> i8 {
  %diff = xls.sub %a, %b: i8
  %umul.5 = xls.umul %diff, %diff : i8
  %umul.6 = xls.umul %c, %diff : i8
  %the_output = xls.add %umul.5, %umul.6 : i8
  return %the_output : i8
}
}
