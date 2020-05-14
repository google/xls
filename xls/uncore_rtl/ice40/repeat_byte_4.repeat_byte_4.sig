module_name: "repeat_byte_4"
data_ports {
  direction: DIRECTION_INPUT
  name: "x"
  width: 8
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out"
  width: 32
}
fixed_latency {
  latency: 1
}
function_type {
  parameters {
    type_enum: BITS
    bit_count: 32
  }
  return_type {
    type_enum: BITS
    bit_count: 32
  }
}
