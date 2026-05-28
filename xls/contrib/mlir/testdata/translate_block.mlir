// RUN: xls_translate --mlir-xls-to-xls %s 2>&1 | FileCheck %s

// CHECK: block passthrough(a: bits[32], out: bits[32])
// CHECK:   a: bits[32] = input_port(name=a
// CHECK:   out: () = output_port(a, name=out
xls.block @passthrough(%a : i32) -> (%out : i32) {
  xls.block_output %a : i32
}
