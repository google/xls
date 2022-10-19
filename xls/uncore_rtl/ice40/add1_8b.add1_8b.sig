module_name: "add1_8b"
data_ports {
  direction: DIRECTION_INPUT
  name: "x"
  width: 8
  type {
    type_enum: BITS
    bit_count: 8
  }
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out"
  width: 8
  type {
    type_enum: BITS
    bit_count: 8
  }
}
fixed_latency {
  latency: 1
}
