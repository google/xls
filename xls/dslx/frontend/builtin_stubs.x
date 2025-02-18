// Copyright 2025 The XLS Authors
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

fn and_reduce<N: u32>(x: uN[N]) -> u1;

fn assert_lt<S: bool, N: u32>(x: xN[S][N], y: xN[S][N]) -> ();

fn assert!<N: u32>(x: bool, msg: u8[N]) -> ();

fn bit_slice_update<N: u32, U: u32, V: u32>(x: uN[N], y: uN[U], z: uN[V]) -> uN[N];

fn clz<N: u32>(x: uN[N]) -> uN[N];

fn cover!<N: u32>(msg: u8[N], y: u1) -> ();

fn ctz<N: u32>(x: uN[N]) -> uN[N];

fn one_hot<N: u32, M: u32={N + 1}>(x: uN[N], y: u1) -> uN[M];

fn one_hot_sel<N: u32, M: u32, S: bool>(x: uN[N], y: xN[S][M][N]) -> xN[S][M];

fn or_reduce<N: u32>(x: uN[N]) -> u1;

fn priority_sel<N: u32, M: u32, S: bool>(x: uN[N], y: xN[S][M][N], z: xN[S][M]) -> xN[S][M];

fn rev<N: u32>(x: uN[N]) -> uN[N];

fn signex<NS: bool, N: u32, MS: bool, M: u32>(x: xN[MS][M], y: xN[NS][N]) -> xN[NS][N];

fn smulp<N: u32>(x: sN[N], y: sN[N]) -> (sN[N], sN[N]);

fn umulp<N: u32>(x: uN[N], y: uN[N]) -> (uN[N], uN[N]);
