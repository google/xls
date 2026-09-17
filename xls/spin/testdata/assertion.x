proc Asserting {
  config() { () }
  init { u32:1 }
  next(state: u32) {
    assert!(state != u32:0, "nonzero");
    state
  }
}
