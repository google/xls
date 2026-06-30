// RUN: xls_translate --mlir-xls-to-xls --split-input-file %s 2>&1 | FileCheck %s

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

// -----

// CHECK: block with_array_reset(clk: clock, rst: bits[1], a: bits[32][2], out: bits[32][2])
// CHECK:   #![reset(port="rst", asynchronous=false, active_low=false)]
// CHECK:   reg state(bits[32][2], reset_value=[0, 42])
// CHECK:   register_write({{.*}}, register=state, reset=rst
xls.block @with_array_reset[clock: "clk", reset: %rst](%a : !xls.array<2 x i32>) -> (%out : !xls.array<2 x i32>) {
  xls.register @state {reset_value = [0 : i32, 42 : i32]} : !xls.array<2 x i32>
  %q = xls.register_read @state : !xls.array<2 x i32>
  xls.register_write @state, %a reset %rst : !xls.array<2 x i32>
  xls.block_output %q : !xls.array<2 x i32>
}

// -----

// CHECK: block with_tuple_reset(clk: clock, rst: bits[1], a: (bits[32], bits[16]), out: (bits[32], bits[16]))
// CHECK:   #![reset(port="rst", asynchronous=false, active_low=false)]
// CHECK:   reg state((bits[32], bits[16]), reset_value=(0, 42))
// CHECK:   register_write({{.*}}, register=state, reset=rst
xls.block @with_tuple_reset[clock: "clk", reset: %rst](%a : tuple<i32, i16>) -> (%out : tuple<i32, i16>) {
  xls.register @state {reset_value = [0 : i32, 42 : i16]} : tuple<i32, i16>
  %q = xls.register_read @state : tuple<i32, i16>
  xls.register_write @state, %a reset %rst : tuple<i32, i16>
  xls.block_output %q : tuple<i32, i16>
}
