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

// Public API header that exposes the XLS `Bits`, `Value`, and value view APIs
// with external visibility.
//
// Though function_builder.h also exposes `Value`s (for use in building literals
// for XLS functions), this is a lighter weight dependency for those consumers
// that don't need function building facilities.

#ifndef XLS_PUBLIC_VALUE_H_
#define XLS_PUBLIC_VALUE_H_

#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"

#endif  // XLS_PUBLIC_VALUE_H_
