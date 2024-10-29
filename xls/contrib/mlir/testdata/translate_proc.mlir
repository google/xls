// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

xls.chan @mychan : i32
xls.chan @vector_chan : !xls.array<32 x i32>

// CHECK: top proc eproc
xls.eproc @eproc(%arg: i32) zeroinitializer {
  xls.yield %arg : i32
}
