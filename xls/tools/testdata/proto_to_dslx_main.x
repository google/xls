struct SubField {
  sub_index: bits[32],
}
struct Field {
  index: bits[32],
  bit_offset: bits[32],
  width: bits[32],
  foo: bits[64][4],
  foo_count: u32,
  sub_fields: SubField[4],
  sub_fields_count: u32,
}
struct Fields {
  fields: Field[5],
  fields_count: u32,
  loner: Field,
}
pub const Foo = Fields { fields: [Field { index: bits[32]:0, bit_offset: bits[32]:0, width: bits[32]:4, foo: [bits[64]:1, bits[64]:2, bits[64]:3, bits[64]:4], foo_count: u32:4, sub_fields: [SubField { sub_index: bits[32]:1 }, SubField { sub_index: bits[32]:2 }, SubField { sub_index: bits[32]:3 }, SubField { sub_index: bits[32]:4 }], sub_fields_count: u32:4 }, Field { index: bits[32]:1, bit_offset: bits[32]:4, width: bits[32]:5, foo: [bits[64]:1, bits[64]:2, bits[64]:3, bits[64]:0], foo_count: u32:3, sub_fields: [SubField { sub_index: bits[32]:1 }, SubField { sub_index: bits[32]:2 }, SubField { sub_index: bits[32]:3 }, SubField { sub_index: bits[32]:0 }], sub_fields_count: u32:3 }, Field { index: bits[32]:2, bit_offset: bits[32]:9, width: bits[32]:1000, foo: [bits[64]:1, bits[64]:2, bits[64]:0, bits[64]:0], foo_count: u32:2, sub_fields: [SubField { sub_index: bits[32]:1 }, SubField { sub_index: bits[32]:2 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }], sub_fields_count: u32:2 }, Field { index: bits[32]:3, bit_offset: bits[32]:1009, width: bits[32]:1, foo: [bits[64]:1, bits[64]:0, bits[64]:0, bits[64]:0], foo_count: u32:1, sub_fields: [SubField { sub_index: bits[32]:1 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }], sub_fields_count: u32:1 }, Field { index: bits[32]:4, bit_offset: bits[32]:1010, width: bits[32]:1, foo: [bits[64]:0, bits[64]:0, bits[64]:0, bits[64]:0], foo_count: u32:0, sub_fields: [SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }], sub_fields_count: u32:0 }], fields_count: u32:5, loner: Field { index: bits[32]:400, bit_offset: bits[32]:400, width: bits[32]:400, foo: [bits[64]:400, bits[64]:0, bits[64]:0, bits[64]:0], foo_count: u32:1, sub_fields: [SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }], sub_fields_count: u32:0 } };