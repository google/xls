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

#include "xls/netlist/fake_cell_library.h"

#include <filesystem>

#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {

absl::StatusOr<CellLibrary> MakeFakeCellLibrary() {
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path proto_path,
      GetXlsRunfilePath("xls/netlist/fake_cell_library.textproto"));

  CellLibraryProto proto;
  XLS_RETURN_IF_ERROR(ParseTextProtoFile(proto_path, &proto));
  return CellLibrary::FromProto(proto);
}

}  // namespace netlist
}  // namespace xls
