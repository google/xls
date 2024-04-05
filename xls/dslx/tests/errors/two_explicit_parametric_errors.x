// Copyright 2024 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This test makes sure that we report the left-most parametric explicit
// instantiation error instead of reporting in a poorly defined order.

fn p<A: u2, B: u3>() -> u3 { A as u3 + B }

fn main() {
  p<u3:0, u4:0>()  // <- two errors in explicit parametric instantiation
}
