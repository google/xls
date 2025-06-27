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

#ifndef XLS_SIMULATION_VERILOG_INCLUDE_H_
#define XLS_SIMULATION_VERILOG_INCLUDE_H_

#include <filesystem>
#include <string>

namespace xls {
namespace verilog {

// Abstraction representing a file tick-included by a Verilog file under
// simulation.
struct VerilogInclude {
  // The relative path of the file in the tick-include statement. For example,
  // this would be "foo/bar.v" for "`include foo/bar.v".
  std::filesystem::path relative_path;

  // The textual Verilog of the included file.
  std::string verilog_text;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_VERILOG_INCLUDE_H_
