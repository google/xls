// Copyright 2020 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPE_AND_PARAMETRIC_ENV_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPE_AND_PARAMETRIC_ENV_H_

#include <memory>

#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Bundles together a type and the parametric environment that was used to
// arrive at that type.
struct TypeAndParametricEnv {
  std::unique_ptr<Type> type;
  ParametricEnv parametric_env;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPE_AND_PARAMETRIC_ENV_H_
