// RUN: xls_opt --xls-lower %s \
// RUN: | xls_translate --mlir-xls-to-xls \
// RUN: > %t

// RUN: eval_ir_main --input 'bits[1]:0; bits[32]:0xab; bits[32]:0xcd' %t \
// RUN: | FileCheck --check-prefix=CHECK-F %s
// RUN: eval_ir_main --input 'bits[1]:1; bits[32]:0xab; bits[32]:0xcd' %t \
// RUN: | FileCheck --check-prefix=CHECK-T %s

// CHECK-F: bits[32]:0xcd
// CHECK-T: bits[32]:0xab

func.func @select(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 attributes { "xls" = true } {
  %0 = "arith.select"(%arg0, %arg1, %arg2) : (i1, i32, i32) -> i32
  return %0 : i32
}