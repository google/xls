// RUN: xls_translate --mlir-xls-to-xls %s 2>&1 | FileCheck %s

// CHECK: block accumulator_le(clk: clock, a: bits[32], le: bits[1], out: bits[32])
// CHECK:   reg state(bits[32])
// CHECK:   register_read(register=state
// CHECK:   add(
// CHECK:   register_write({{.*}}, register=state, load_enable=le
// CHECK:   output_port({{.*}}, name=out
xls.block @accumulator_le[clock: "clk"](%a : i32, %le : i1) -> (%out : i32) {
  xls.register @state : i32
  %q = xls.register_read @state : i32
  %sum = xls.add %a, %q : i32
  xls.register_write @state, %sum load_enable %le : i32
  xls.block_output %q : i32
}
