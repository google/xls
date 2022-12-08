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

// Public API header that exposes the XLS IR entity APIs (e.g. Packages,
// Functions) with external visibility.
//
// TODO(leary): 2022-10-26 Restrict the interfaces on IR types that we expose
// for use by external clients to those that are more likely to remain stable /
// most often have public use cases.

#ifndef XLS_PUBLIC_IR_H_
#define XLS_PUBLIC_IR_H_

#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/package.h"

namespace xls {

// Given a function "f" returns the package that owns it.
//
// Note: all XLS functions reside within a package.
const Package& GetPackage(const Function& f);

}  // namespace xls

#endif  // XLS_PUBLIC_IR_H_
