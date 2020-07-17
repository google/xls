fn [M: u32, N: u32, R: u32 = M + N] smul_generic(x: sN[M], y: sN[N]) -> sN[R] {
  (x as sN[R]) * (y as sN[R])
}

fn smul_s2_s3(x: s2, y: s3) -> s5 {
  smul_generic(x, y)
}

fn smul_s3_s4(x: s3, y: s4) -> s7 {
  smul_generic(x, y)
}

test parametric_smul {
  let _ = assert_eq(s5:2, smul_s2_s3(s2:-1, s3:-2)) in
  let _ = assert_eq(s5:6, smul_s2_s3(s2:-2, s3:-3)) in
  let _ = assert_eq(s5:-6, smul_s2_s3(s2:-2, s3:3)) in
  let _ = assert_eq(s7:-7, smul_s3_s4(s3:-1, s4:7)) in
  ()
}
