// RUN: xls_translate --mlir-xls-to-xls --main-function=parent_block %s 2>&1 | FileCheck %s

// CHECK: block child_block(in: bits[32], out: bits[32])
// CHECK:   input_port(name=in
// CHECK:   output_port({{.*}}, name=out

// CHECK: block parent_block(
// CHECK:   instantiation child_inst(block=child_block, kind=block)
// CHECK:   instantiation my_fifo(data_type=bits[32], depth=10, bypass=false, register_push_outputs=true, register_pop_outputs=true, kind=fifo)
// CHECK:   input_port(name=rst
// CHECK:   input_port(name=x
// CHECK:   instantiation_input({{.*}}, instantiation=my_fifo, port_name=push_data
// CHECK:   instantiation_input({{.*}}, instantiation=my_fifo, port_name=push_valid
// CHECK:   instantiation_input({{.*}}, instantiation=my_fifo, port_name=pop_ready
// CHECK:   instantiation_input({{.*}}, instantiation=my_fifo, port_name=rst
// CHECK:   instantiation_output(instantiation=my_fifo, port_name=push_ready
// CHECK:   instantiation_output(instantiation=my_fifo, port_name=pop_data
// CHECK:   instantiation_output(instantiation=my_fifo, port_name=pop_valid
// CHECK:   output_port({{.*}}, name=y
xls.block @child_block(%in : i32) -> (%out : i32) {
  xls.block_output %in : i32
}
xls.block @parent_block[clock: "clk", reset: %rst](%x : i32, %pv : i1, %pr : i1) -> (%y : i32) {
  %child_out = xls.block_instantiate "child_inst" @child_block(%x) : (i32) -> (i32)
  %push_ready, %pop_data, %pop_valid =
    xls.fifo_instantiate "my_fifo"(%child_out, %pv, %pr, %rst)
      attributes {depth = 10 : i64, bypass = false, register_push_outputs = true,
       register_pop_outputs = true}
      : (i32, i1, i1, i1) -> (i1, i32, i1)
  xls.block_output %pop_data : i32
}
