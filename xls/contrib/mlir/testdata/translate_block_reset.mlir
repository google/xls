// RUN: xls_translate --mlir-xls-to-xls %s 2>&1 | FileCheck %s

// CHECK: block with_reset(clk: clock, rst: bits[1], a: bits[32], out: bits[32])
// CHECK:   #![reset(port="rst", asynchronous=false, active_low=false)]
// CHECK:   reg state(bits[32], reset_value=0)
// CHECK:   register_write({{.*}}, register=state, reset=rst
xls.block @with_reset[clock: "clk", reset: %rst](%a : i32) -> (%out : i32) {
  xls.register @state {reset_value = 0 : i32} : i32
  %q = xls.register_read @state : i32
  %sum = xls.add %a, %q : i32
  xls.register_write @state, %sum reset %rst : i32
  xls.block_output %q : i32
}
