struct Field {
  index: bits[32],
  bit_offset: bits[32],
  width: bits[32],
  foo: bits[64][4],
  foo_count: u32,
}

struct Fields {
  fields: Field[4],
  fields_count: u32,
  loner: Field,
}

pub fn Foo() -> Fields {
  let tmp: Fields =   Fields {
    fields: Field[4]:[
      Field {
        index: bits[32]: 0x0,
        bit_offset: bits[32]: 0x0,
        width: bits[32]: 0x4,
        foo: bits[64][4]:[bits[64]:1, bits[64]:2, bits[64]:3, bits[64]:4],
        foo_count: bits[32]:0x4
      },
      Field {
        index: bits[32]: 0x1,
        bit_offset: bits[32]: 0x4,
        width: bits[32]: 0x5,
        foo: bits[64][4]:[bits[64]:1, bits[64]:2, bits[64]:3, ...],
        foo_count: bits[32]:0x3
      },
      Field {
        index: bits[32]: 0x2,
        bit_offset: bits[32]: 0x9,
        width: bits[32]: 0x3e8,
        foo: bits[64][4]:[bits[64]:1, bits[64]:2, ...],
        foo_count: bits[32]:0x2
      },
      Field {
        index: bits[32]: 0x3,
        bit_offset: bits[32]: 0x3f1,
        width: bits[32]: 0x1,
        foo: bits[64][4]:[bits[64]:1, ...],
        foo_count: bits[32]:0x1
      }
    ],
    fields_count: u32:0x4,
    loner:    Field {
      index: bits[32]: 0x190,
      bit_offset: bits[32]: 0x190,
      width: bits[32]: 0x190,
      foo: bits[64][4]:[bits[64]:400, ...],
      foo_count: bits[32]:0x1
    }
  };
  tmp
}
