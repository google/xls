// RUN: xls_translate --mlir-xls-to-xls --main-function=parent_block %s 2>&1 | FileCheck %s
// RUN: xls_translate --mlir-xls-to-xls --main-function=parent_block_reset %s 2>&1 | FileCheck %s --check-prefix=RESET

// CHECK: block child_block(in: bits[32], out: bits[32])
// CHECK:   input_port(name=in
// CHECK:   output_port({{.*}}, name=out

// CHECK: block parent_block(x: bits[32], y: bits[32])
// CHECK:   instantiation child_inst(block=child_block, kind=block)
// CHECK:   input_port(name=x
// CHECK:   instantiation_input({{.*}}, instantiation=child_inst, port_name=in
// CHECK:   instantiation_output(instantiation=child_inst, port_name=out
// CHECK:   output_port({{.*}}, name=y
xls.block @child_block(%in : i32) -> (%out : i32) {
  xls.block_output %in : i32
}
xls.block @parent_block(%x : i32) -> (%y : i32) {
  %result = xls.block_instantiate "child_inst" @child_block(%x) : (i32) -> (i32)
  xls.block_output %result : i32
}

// RESET: block child_block_reset(clk: clock, rst: bits[1], in: bits[32], out: bits[32])
// RESET:   input_port(name=rst
// RESET:   input_port(name=in
// RESET:   output_port({{.*}}, name=out

// RESET: block parent_block_reset(clk: clock, rst: bits[1], x: bits[32], y: bits[32])
// RESET:   instantiation child_inst(block=child_block_reset, kind=block)
// RESET:   input_port(name=rst
// RESET:   input_port(name=x
// RESET:   instantiation_input({{.*}}, instantiation=child_inst, port_name=rst
// RESET:   instantiation_input({{.*}}, instantiation=child_inst, port_name=in
// RESET:   instantiation_output(instantiation=child_inst, port_name=out
// RESET:   output_port({{.*}}, name=y

xls.block @child_block_reset[clock: "clk", reset: %rst](%in : i32) -> (%out : i32) {
  xls.block_output %in : i32
}
xls.block @parent_block_reset[clock: "clk", reset: %rst](%x : i32) -> (%y : i32) {
  %result = xls.block_instantiate "child_inst" @child_block_reset[reset: %rst](%x) : (i32) -> (i32)
  xls.block_output %result : i32
}
