test comparisons {
  let _: () = assert_eq(true, u32:2 < u32:3) in
  let _: () = assert_eq(true, u32:2 <= u32:3) in
  let _: () = assert_eq(false, u32:2 > u32:3) in
  let _: () = assert_eq(false, u32:2 >= u32:3) in
  let _: () = assert_eq(false, u32:2 == u32:3) in
  let _: () = assert_eq(true, u32:2 != u32:3) in
  ()
}
