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

#ifndef XLS_TOOLS_EXTRACT_INTERFACE_H_
#define XLS_TOOLS_EXTRACT_INTERFACE_H_

#include "xls/ir/package.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls {

// Create a PackageInterfaceProto based on the given Package.
PackageInterfaceProto ExtractPackageInterface(Package* package);

}  // namespace xls

#endif  // XLS_TOOLS_EXTRACT_INTERFACE_H_
