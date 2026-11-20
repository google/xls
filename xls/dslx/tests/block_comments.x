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

/* Various tests regarding comments. */

/* Can comments contain non-ASCII characters: ∞ */

// empty comments are allowed
/**/

// can start from any column

      /* */

// empty comments spanning multiple lines

/*

*/

/*consecutive comments*//*without whitespace*/

/*including*///line comments

// Something more real
/** main(): does nothing

but it is still important to us

 */

// block comments containing line comments are fine

/*
// This comment is controversial, so it has been nested
*/

// Dummy "main" function to enable use of the regular test macros.

fn main() { () }
