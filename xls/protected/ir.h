// Copyright 2022 The XLS Authors
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

#ifndef XLS_PROTECTED_IR_H_
#define XLS_PROTECTED_IR_H_

// This file contains APIs for working more with "the guts" of the IR.
//
// It is more than the public API offers, and is subject to change, so we put it
// in the "protected" directory.
//
// These details are all subject to change.

// IWYU pragma: begin_exports
#include "xls/ir/function.h"
#include "xls/ir/topo_sort.h"
// IWYU pragma: end_exports

#endif  // XLS_PROTECTED_IR_H_
