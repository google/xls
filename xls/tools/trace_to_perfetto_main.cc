// Copyright 2025 The XLS Authors
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

#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "protos/perfetto/trace/trace.pb.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/interpreter/trace.pb.h"
#include "xls/tools/trace_to_perfetto.h"

ABSL_FLAG(std::string, input_trace, "",
          "Path to the input trace file (riegeli format).");
ABSL_FLAG(std::string, output_trace, "",
          "Path to the output perfetto.protos.Trace file.");

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);

  std::string input_path = absl::GetFlag(FLAGS_input_trace);
  QCHECK(!input_path.empty()) << "--input_trace is required.";
  std::string output_path = absl::GetFlag(FLAGS_output_trace);
  QCHECK(!output_path.empty()) << "--output_trace is required.";

  riegeli::RecordReader reader{riegeli::FdReader(input_path)};
  CHECK_OK(reader.status());
  absl::StatusOr<perfetto::protos::Trace> perfetto_trace =
      xls::TraceToPerfetto(reader);
  QCHECK(reader.Close());
  QCHECK_OK(perfetto_trace.status());

  QCHECK_OK(xls::SetProtobinFile(output_path, *perfetto_trace));

  std::cout << "Perfetto trace written to " << output_path << std::endl;

  return 0;
}
