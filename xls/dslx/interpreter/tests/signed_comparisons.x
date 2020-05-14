test signed_comparisons {
  let _: () = assert_eq(true,  slt(u32:2, u32:3)) in
  let _: () = assert_eq(true,  sle(u32:2, u32:3)) in
  let _: () = assert_eq(false, sgt(u32:2, u32:3)) in
  let _: () = assert_eq(false, sge(u32:2, u32:3)) in

  // Mixed positive and negative numbers.
  let _: () = assert_eq(true,  slt(u32:2, u32:3)) in
  let _: () = assert_eq(true,  sle(u32:2, u32:3)) in
  let _: () = assert_eq(false, sgt(u32:2, u32:3)) in
  let _: () = assert_eq(false, sge(u32:2, u32:3)) in

  // Negative vs negative numbers.
  let _: () = assert_eq(false, slt(u32:-2, u32:-3)) in
  let _: () = assert_eq(true,  slt(u32:-3, u32:-2)) in
  let _: () = assert_eq(false, slt(u32:-3, u32:-3)) in

  let _: () = assert_eq(false, sle(u32:-2, u32:-3)) in
  let _: () = assert_eq(true,  sle(u32:-3, u32:-2)) in

  let _: () = assert_eq(true,  sgt(u32:-2, u32:-3)) in
  let _: () = assert_eq(false, sgt(u32:-2, u32:-2)) in
  let _: () = assert_eq(false, sgt(u32:-3, u32:-2)) in

  let _: () = assert_eq(false, sge(u32:-3, u32:-2)) in
  let _: () = assert_eq(true, sge(u32:-2, u32:-3)) in
  let _: () = assert_eq(true, sge(u32:-3, u32:-3)) in
  ()
}
