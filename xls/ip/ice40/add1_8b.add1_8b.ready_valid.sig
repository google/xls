module_name: "add1_8b"
data_ports {
  direction: DIRECTION_INPUT
  name: "x"
  width: 8
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out"
  width: 8
}
clock_name: "clk"
ready_valid {
  input_ready: "input_ready"
  input_valid: "input_valid"
  output_ready: "out_ready"
  output_valid: "out_valid"
}
function_type {
  parameters {
    type_enum: BITS
    bit_count: 8
  }
  return_type {
    type_enum: BITS
    bit_count: 8
  }
}
