// RUN: xls_opt --arith-to-xls --xls-lower --canonicalize %s | \
// RUN:  xls_translate --mlir-xls-to-xls --main-function=add16 -o %t
// RUN: eval_ir_main --input '[(bits[1]:0x0, bits[8]:0x7e, bits[7]:0x0), (bits[1]:0x0, bits[8]:0x7f, bits[7]:0x0)]; [(bits[1]:0x0, bits[8]:0x7e, bits[7]:0x0), (bits[1]:0x0, bits[8]:0x7f, bits[7]:0x0)]' --top=add16 %t | \
// RUN:   FileCheck %s

func.func @add16(%arg0: tensor<2xbf16>, %arg1: tensor<2xbf16>) -> (tensor<2xbf16>) attributes { "xls" = true } {
  // CHECK: [(bits[1]:0x0, bits[8]:0x7f, bits[7]:0x0), (bits[1]:0x0, bits[8]:0x80, bits[7]:0x0)]
  %0 = arith.addf %arg0, %arg1: tensor<2xbf16>
  return %0 : tensor<2xbf16>
}
