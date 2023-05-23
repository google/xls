// Copyright 2023 The XLS Authors
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

#ifndef XLS_IR_FOREIGN_FUNCTION_H_
#define XLS_IR_FOREIGN_FUNCTION_H_

#include <string>

namespace xls {

// Meta Information about a foreign function call.
// Right now, this is just the name, but this can include additional information
// such as call conventions, parameters, and return values.
struct ForeignFunctionData {
  std::string name;
};

}  // namespace xls

#endif  // XLS_IR_FOREIGN_FUNCTION_H_
