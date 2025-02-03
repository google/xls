// RUN: xls_opt --xls-lower %s | xls_translate --mlir-xls-to-xls -o %t && \
// RUN: eval_ir_main --input '[bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0, bits[32]:0]; bits[32]:0xab' %t | \
// RUN:  FileCheck --check-prefix=CHECK %s

func.func @tensor_insert(%arg0: !xls.array<14 x i32>, %arg1: i32) -> !xls.array<14 x i32> attributes {xls = true} {
  %c_10 = "xls.constant_scalar"() <{value = 10 : index}> : () -> index
  %4 = "xls.array_update"(%arg0, %arg1, %c_10) : (!xls.array<14 x i32>, i32, index) -> !xls.array<14 x i32>
  return %4 : !xls.array<14 x i32>
}
// Check value is correctly inserted into array.
//                             0,            1,            2,            3,            4,            5,            6,            7,            8,            9,           10,           11,           12,            13
// CHECK{LITERAL}: [bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0, bits[32]:0xab, bits[32]:0x0, bits[32]:0x0, bits[32]:0x0]
