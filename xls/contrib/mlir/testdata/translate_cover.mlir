// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 \
// RUN:   | FileCheck %s --dump-input-filter=all

// CHECK-LABEL: top fn test_cover(
// CHECK-SAME: [[ARG0:.*]]: bits[1]
// CHECK: cover([[ARG0]], label="my_coverpoint"
func.func @test_cover(%arg0: i1) {
  xls.cover %arg0, "my_coverpoint"
  func.return
}
