// Copyright 2021 The XLS Authors
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

module ifte(i, t, e, out);
  input [7:0] e;
  input i;
  output [7:0] out;
  input [7:0] t;
  imux2 _0_ (
    .A(t[0]),
    .B(e[0]),
    .S(i),
    .Y(out[0])
  );
  imux2 _1_ (
    .A(t[1]),
    .B(e[1]),
    .S(i),
    .Y(out[1])
  );
  imux2 _2_ (
    .A(t[2]),
    .B(e[2]),
    .S(i),
    .Y(out[2])
  );
  imux2 _3_ (
    .A(t[3]),
    .B(e[3]),
    .S(i),
    .Y(out[3])
  );
  imux2 _4_ (
    .A(t[4]),
    .B(e[4]),
    .S(i),
    .Y(out[4])
  );
  imux2 _5_ (
    .A(t[5]),
    .B(e[5]),
    .S(i),
    .Y(out[5])
  );
  imux2 _6_ (
    .A(t[6]),
    .B(e[6]),
    .S(i),
    .Y(out[6])
  );
  imux2 _7_ (
    .A(t[7]),
    .B(e[7]),
    .S(i),
    .Y(out[7])
  );
endmodule
