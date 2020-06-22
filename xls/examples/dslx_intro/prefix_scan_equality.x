// Prefix scans an array of 8 32-bit values and produces a running count of
// duplicate values in the run.
fn prefix_scan_eq(x: u32[8]) -> u3[8] {
  let (_, _, result) =
    for ((i, elem), (prior, count, result)): ((u32, u32), (u32, u3, u3[8]))
          in enumerate(x) {
    let (to_place, new_count): (u3, u3) = match (i == u32:0, prior == elem) {
      // The first iteration always places 0 and propagates seen count of 1.
      (true, _) => (u3:0, u3:1);
      // Subsequent iterations propagate seen count of previous_seen_count+1 if
      // the current element matches the prior one, and places the current seen
      // count.
      (false, true) => (count, count + u3:1);
      // If the current element doesn't match the prior one we propagate a seen
      // count of 1 and place a seen count of 0.
      (false, false) => (u3:0, u3:1);
    } in
    let new_result: u3[8] = update(result, i, to_place) in
    (elem, new_count, new_result)
  }((u32:-1, u3:0, u3[8]:[u3:0, ...])) in
  result
}

test prefix_scan_eq_all_zero {
  let input = u32[8]:[0, ...] in
  let result = prefix_scan_eq(input) in
  assert_eq(result, u3[8]:[0, 1, 2, 3, 4, 5, 6, 7])
}

test prefix_scan_eq_doubles {
  let input = u32[8]:[0, 0, 1, 1, 2, 2, 3, 3] in
  let result = prefix_scan_eq(input) in
  assert_eq(result, u3[8]:[0, 1, 0, 1, 0, 1, 0, 1])
}
