module_name: "repeat_byte_4"
data_ports {
  direction: PORT_DIRECTION_INPUT
  name: "x"
  width: 8
  type {
    type_enum: BITS
    bit_count: 8
  }
}
data_ports {
  direction: PORT_DIRECTION_OUTPUT
  name: "out"
  width: 32
  type {
    type_enum: BITS
    bit_count: 32
  }
}
fixed_latency {
  latency: 1
}
