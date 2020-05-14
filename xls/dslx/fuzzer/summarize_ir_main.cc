// Copyright 2020 Google LLC
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

#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/fuzzer/sample_summary.pb.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

const char kUsage[] = R"(
Appends a summary of a given IR to the specified Protobuf summary
file. emitted. The summary information includes information such as op types,
widths, etc. This is used by the fuzzer to give an indication of what kind of IR
operations are being covered. Usage:

  summarize_ir_main --summary_file=SUMMARY_FILE IR_FILE

The summary file will be created if it does not exist. Otherwise the summary
file is appended to.
)";

ABSL_FLAG(std::string, tag, "",
          "String describing the provenance of the sample. Example: "
          "\"before-opt\".");
ABSL_FLAG(std::string, summary_file, "", "Summary file to append to.");

namespace xls {
namespace {

std::string TypeToString(Type* type) {
  if (type->IsBits()) {
    return "bits";
  } else if (type->IsArray()) {
    return "array";
  } else if (type->IsTuple()) {
    return "tuple";
  }
  return "other";
}

fuzzer::SampleSummaries SummarizePackage(Package* package) {
  fuzzer::SampleSummaries summaries;
  fuzzer::SampleSummaryProto* summary = summaries.add_samples();
  summary->set_tag(absl::GetFlag(FLAGS_tag));
  for (const auto& function : package->functions()) {
    for (Node* node : function->nodes()) {
      fuzzer::NodeProto* node_proto = summary->add_nodes();
      node_proto->set_op(OpToString(node->op()));
      node_proto->set_type(TypeToString(node->GetType()));
      node_proto->set_width(node->GetType()->GetFlatBitCount());
      for (Node* operand : node->operands()) {
        fuzzer::NodeProto* operand_proto = node_proto->add_operands();
        operand_proto->set_op(OpToString(operand->op()));
        operand_proto->set_type(TypeToString(operand->GetType()));
        operand_proto->set_width(operand->GetType()->GetFlatBitCount());
      }
    }
  }
  return summaries;
}

absl::Status RealMain(absl::string_view input_path) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(input_path));
  std::unique_ptr<Package> package;
  XLS_ASSIGN_OR_RETURN(package, Parser::ParsePackage(contents, input_path));

  fuzzer::SampleSummaries summaries = SummarizePackage(package.get());

  XLS_QCHECK(!absl::GetFlag(FLAGS_summary_file).empty())
      << "Must specify --summary_file.";
  // Since SampleSummaries contains just a repeated field, appending a new
  // such proto to a file that could potentially contain a SampleSummaries
  // already will make the file contain a valid SampleSummaries proto where
  // the prior repeated field and the one of `summaries` is concatenated. This
  // is essentially a quick and dirty RecordIO/Rigeli without the compression,
  // seeking and corruption handling features.
  XLS_RETURN_IF_ERROR(AppendStringToFile(absl::GetFlag(FLAGS_summary_file),
                                         summaries.SerializeAsString()));

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.empty()) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s IR_FILE",
                                          argv[0]);
  }

  XLS_QCHECK_OK(xls::RealMain(positional_arguments[0]));
  return EXIT_SUCCESS;
}
