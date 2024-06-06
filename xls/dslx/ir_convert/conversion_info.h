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

#ifndef XLS_DSLX_IR_CONVERT_CONVERSION_INFO_H_
#define XLS_DSLX_IR_CONVERT_CONVERSION_INFO_H_

#include <memory>
#include <string>

#include "xls/ir/package.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {

// Container of all data obtained while converting a DSLX design into a package.
struct PackageConversionData {
  // The package IR representation.
  std::unique_ptr<Package> package;
  // Any extern type/interface information
  PackageInterfaceProto interface;

  std::string DumpIr() const { return package->DumpIr(); }
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_CONVERSION_INFO_H_
