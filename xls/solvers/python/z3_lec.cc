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

#include "xls/solvers/z3_lec.h"

#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/import_status_module.h"
#include "xls/ir/ir_parser.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist.pb.h"
#include "xls/netlist/netlist_parser.h"

namespace py = pybind11;

namespace xls {
namespace solvers {

PYBIND11_MODULE(z3_lec, m) {
  ImportStatusModule();

  // This takes a textproto instead of a fully realized protocol buffer, as
  // there's no pybind11 caster for protobufs, and writing one isn't a great
  // idea because A) there's supposedly an internal one being open-sourced soon,
  // and B) it'd be a big ol' yak to shave. When there's a public protobuf
  // caster, this should migrate onto it.
  m.def(
      "run",
      [](std::string_view ir_text, std::string_view netlist_text,
         std::string_view netlist_module_name,
         std::string_view cell_library_textproto) -> absl::StatusOr<bool> {
        XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));

        netlist::rtl::Scanner scanner(netlist_text);
        netlist::CellLibraryProto proto;
        google::protobuf::TextFormat::ParseFromString(std::string(cell_library_textproto),
                                            &proto);
        XLS_ASSIGN_OR_RETURN(auto cell_library,
                             netlist::CellLibrary::FromProto(proto));
        XLS_ASSIGN_OR_RETURN(auto netlist, netlist::rtl::Parser::ParseNetlist(
                                               &cell_library, &scanner));

        z3::LecParams params;
        params.ir_package = package.get();
        XLS_ASSIGN_OR_RETURN(auto entry_function, package->GetTopAsFunction());
        params.ir_function = entry_function;
        params.netlist = netlist.get();
        params.netlist_module_name = netlist_module_name;

        XLS_ASSIGN_OR_RETURN(auto lec, z3::Lec::Create(params));
        return lec->Run();
      },
      py::arg("ir_text"), py::arg("netlist_text"),
      py::arg("netlist_module_name"), py::arg("cell_library_textproto"));
}

}  // namespace solvers
}  // namespace xls
