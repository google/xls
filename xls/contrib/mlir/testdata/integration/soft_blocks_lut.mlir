// RUN: xls/contrib/mlir/xls_opt --xls-lower %s \
// RUN: | xls/contrib/mlir/xls_translate --mlir-xls-to-xls \
// RUN: > %t

// RUN: xls/contrib/mlir/xls_opt --xls-lower %s \
// RUN: | xls/contrib/mlir/xls_translate --mlir-xls-to-verilog -- --delay_model=asap7 --generator=combinational \
// RUN: | FileCheck --check-prefix=CHECK-VERILOG %s

// RUN: xls/tools/codegen_main %t \
// RUN:   --output_verilog_path %t.v --generator=combinational \
// RUN: && FileCheck --check-prefix=CHECK-VERILOG %s < %t.v

// CHECK-VERILOG: module {{.+}}(
// CHECK-VERILOG: endmodule

// RUN: xls/tools/eval_ir_main --input '[bits[1]:0, bits[1]:0]' %t \
// RUN: | FileCheck --check-prefix=CHECK-FF %s

// RUN: xls/tools/eval_ir_main --input '[bits[1]:1, bits[1]:0]' %t \
// RUN: | FileCheck --check-prefix=CHECK-TF %s

// RUN: xls/tools/eval_ir_main --input '[bits[1]:0, bits[1]:1]' %t \
// RUN: | FileCheck --check-prefix=CHECK-FT %s

// RUN: xls/tools/eval_ir_main --input '[bits[1]:1, bits[1]:1]' %t \
// RUN: | FileCheck --check-prefix=CHECK-TT %s

// CHECK-FF: [bits[1]:0x1, bits[1]:0x0, bits[1]:0x0, bits[1]:0x0]
// CHECK-TF: [bits[1]:0x0, bits[1]:0x1, bits[1]:0x0, bits[1]:0x0]
// CHECK-FT: [bits[1]:0x0, bits[1]:0x0, bits[1]:0x1, bits[1]:0x0]
// CHECK-TT: [bits[1]:0x0, bits[1]:0x0, bits[1]:0x0, bits[1]:0x1]

func.func @test_func(%arg0: tensor<2xi1>) -> tensor<1x2x2xi1> attributes {llvm.emit_c_interface, xls} {
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<2xi1>
  %c1 = arith.constant 1 : index
  %extracted_0 = tensor.extract %arg0[%c1] : tensor<2xi1>
  %cst = arith.constant dense<[[true, false, false, false], [false, true, false, false], [false, false, true, false], [false, false, false, true]]> : tensor<4x4xi1>
  %c0_1 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %0 = arith.extui %extracted : i1 to i64
  %1 = arith.index_cast %0 : i64 to index
  %2 = arith.shli %1, %c0_2 : index
  %3 = arith.addi %c0_1, %2 : index
  %c1_3 = arith.constant 1 : index
  %4 = arith.extui %extracted_0 : i1 to i64
  %5 = arith.index_cast %4 : i64 to index
  %6 = arith.shli %5, %c1_3 : index
  %7 = arith.addi %3, %6 : index
  %extracted_slice = tensor.extract_slice %cst[%7, 0] [1, 4] [1, 1] : tensor<4x4xi1> to tensor<4xi1>
  %c0_4 = arith.constant 0 : index
  %extracted_5 = tensor.extract %extracted_slice[%c0_4] : tensor<4xi1>
  %c1_6 = arith.constant 1 : index
  %extracted_7 = tensor.extract %extracted_slice[%c1_6] : tensor<4xi1>
  %c2 = arith.constant 2 : index
  %extracted_8 = tensor.extract %extracted_slice[%c2] : tensor<4xi1>
  %c3 = arith.constant 3 : index
  %extracted_9 = tensor.extract %extracted_slice[%c3] : tensor<4xi1>
  %from_elements = tensor.from_elements %extracted_5, %extracted_7, %extracted_8, %extracted_9 : tensor<1x2x2xi1>
  return %from_elements : tensor<1x2x2xi1>
}
