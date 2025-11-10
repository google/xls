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

#ifndef XLS_PUBLIC_RUNTIME_BUILD_ACTIONS_H_
#define XLS_PUBLIC_RUNTIME_BUILD_ACTIONS_H_

// Exposes XLS functionality at the level of "build actions" (e.g. the kinds of
// things we specify in Bazel BUILD rules), so they can be invoked at runtime by
// XLS consumers who want to exercise Just-in-Time capabilities, e.g. for users
// sweeping a parameterized design space at runtime.
//
// These APIs attempt to be minimal in both their surface area (number of
// routines) and their options (number of parameters), so they can remain mostly
// stable in their signatures, so long as the basic functionality exists.
//
// Note that, just like users should not depending on the precise output of a
// compiler remaining stable, users should not depend on the precise output of
// these actions remaining stable, they will evolve as the XLS system evolves.

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/public/runtime_codegen_actions.h"  // IWYU pragma: export
#include "xls/public/runtime_dslx_actions.h"  // IWYU pragma: export
#include "xls/public/runtime_ir_opt_actions.h"  // IWYU pragma: export

#endif  // XLS_PUBLIC_RUNTIME_BUILD_ACTIONS_H_
