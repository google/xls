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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/estimators/delay_model/sample_point_extraction_utils.h"
#include "xls/estimators/estimator_model.pb.h"
#include "xls/ir/ir_parser.h"

namespace {

constexpr std::string_view kUsage = R"(
Extracts delay model sample points that would be used by an actual design.

Example invocation:

  extract_sample_points_from_ir_main --op_models_path=op_models.textproto --ir_path=foo.ir
)";

// How to sort operations when applying the `limit` command line flag.
enum class LimitType : uint8_t { kFrequency, kDelay };

inline bool AbslParseFlag(std::string_view text, LimitType* out,
                          std::string* error) {
  if (text == "frequency") {
    *out = LimitType::kFrequency;
    return true;
  }
  if (text == "delay") {
    *out = LimitType::kDelay;
    return true;
  }
  *error = "Unrecognized limit type.";
  return false;
}

inline std::string AbslUnparseFlag(LimitType limit) {
  switch (limit) {
    case LimitType::kFrequency:
      return "frequency";
    case LimitType::kDelay:
      return "delay";
    default:
      return "unknown";
  }
}

}  // namespace

ABSL_FLAG(std::string, ir_path, "", "Path to the IR file to load.");

ABSL_FLAG(std::string, op_models_path, "",
          "Path to either an `OpModels` or `DelayModel` text proto file, for "
          "determining which types of operations can use data points.");

ABSL_FLAG(std::string, out_path, "",
          "Path to the `OpSamplesList` text proto to generate, if desired. "
          "Otherwise, it is printed to standard output.");

ABSL_FLAG(size_t, limit, std::numeric_limits<size_t>::max(),
          "Limits the number of sample points besides the required kIdentity "
          "sample.");

ABSL_FLAG(LimitType, limit_type, LimitType::kFrequency,
          "How to apply the --limit flag. Either the most-frequent or the "
          "highest-delay operations can be selected.");

ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from the registry. This is required if "
          "--limit_type is `delay`. Otherwise, it just causes optional delay "
          "info to be displayed.");

namespace xls {
namespace {

// Loads an `OpModels` proto from the text proto file in `path`. The file should
// contain either an `OpModels` message or a `DelayModel` message wrapping one.
absl::StatusOr<estimator_model::OpModels> LoadOpModels(std::string_view path) {
  XLS_ASSIGN_OR_RETURN(std::string content, GetFileContents(path));
  estimator_model::OpModels models;
  bool success = google::protobuf::TextFormat::ParseFromString(content, &models);
  if (!success) {
    // The `op_models_path` may actually point to a complete delay model,
    // containing op models plus other artifacts.
    models.Clear();
    estimator_model::EstimatorModel delay_model;
    success = google::protobuf::TextFormat::ParseFromString(content, &delay_model);
    if (delay_model.metric() != estimator_model::Metric::DELAY_METRIC) {
      return absl::UnimplementedError(
          "extract_sample_points_from_ir_main only supports delay model");
    }
    if (!success) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Not a valid OpModels or EstimatorModel text proto file: ", path));
    }
    *models.mutable_op_models() = std::move(*delay_model.mutable_op_models());
  }
  return models;
}

// Converts a `Parameterization` message to a concise string suitable for one
// cell in the table this program writes to standard output.
std::string ParamsToString(const estimator_model::Parameterization& params) {
  std::string out;
  absl::StrAppend(&out, absl::StrJoin(params.operand_widths(), ", "));
  if (!out.empty()) {
    absl::StrAppend(&out, " -> ");
  }
  absl::StrAppend(&out, params.result_width());
  return out;
}

// Sorts the given `points` in descending order by a particular field, and also
// prints a table in that order to standard error. The field of interest is
// accessed by `accessor` and labeled in the table with `field_title`.
void SortDescendingAndPrint(
    std::vector<estimator_model::SamplePoint>& points,
    std::string_view field_title,
    absl::AnyInvocable<int64_t(const estimator_model::SamplePoint&)> accessor) {
  absl::c_stable_sort(points, [&](const estimator_model::SamplePoint& x,
                                  const estimator_model::SamplePoint& y) {
    return accessor(x) > accessor(y);
  });
  std::cerr << absl::StreamFormat("%-25s %-40s %12s\n", "Operation", "Params",
                                  field_title);
  for (const estimator_model::SamplePoint& point : points) {
    std::cerr << absl::StreamFormat("%-25s %-40s %12d\n", point.op_name,
                                    ParamsToString(point.params),
                                    accessor(point));
  }
  std::cerr << "\n";
}

absl::Status RealMain() {
  const std::string ir_path = absl::GetFlag(FLAGS_ir_path);
  const std::string op_models_path = absl::GetFlag(FLAGS_op_models_path);
  const std::string delay_model = absl::GetFlag(FLAGS_delay_model);
  const std::string out_path = absl::GetFlag(FLAGS_out_path);
  const size_t limit = absl::GetFlag(FLAGS_limit);
  const LimitType limit_type = absl::GetFlag(FLAGS_limit_type);
  if (ir_path.empty()) {
    return absl::InvalidArgumentError("--ir_path must be specified.");
  }
  if (op_models_path.empty()) {
    return absl::InvalidArgumentError("--op_models_path must be specified.");
  }
  if (delay_model.empty() && limit_type == LimitType::kDelay) {
    return absl::InvalidArgumentError(
        "--delay_model must be specified when using --limit_type=delay.");
  }
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(estimator_model::OpModels op_models,
                       LoadOpModels(op_models_path));
  std::optional<DelayEstimator*> delay_estimator;
  if (!delay_model.empty()) {
    XLS_ASSIGN_OR_RETURN(delay_estimator, GetDelayEstimator(delay_model));
  }
  XLS_ASSIGN_OR_RETURN(
      std::vector<estimator_model::SamplePoint> points,
      ExtractSamplePoints(*package, op_models, delay_estimator));

  // Print a table in possibly two different orders, using the order specified
  // by `--limit_type` to construct the list proto, which is the last output.
  // Note that the order of `points` passed to `ConvertToOpSamplesList` only
  // affects its output if `--limit` is specified.
  SortDescendingAndPrint(
      points, "Frequency",
      [](const estimator_model::SamplePoint& x) { return x.frequency; });
  estimator_model::OpSamplesList list;
  if (limit_type == LimitType::kFrequency) {
    list = ConvertToOpSamplesList(points, limit);
  }
  if (delay_estimator.has_value()) {
    SortDescendingAndPrint(points, "Delay",
                           [](const estimator_model::SamplePoint& x) {
                             return x.delay_estimate_in_ps;
                           });
    if (limit_type == LimitType::kDelay) {
      list = ConvertToOpSamplesList(points, limit);
    }
  }
  std::string text_proto;
  google::protobuf::TextFormat::PrintToString(list, &text_proto);
  if (!out_path.empty()) {
    XLS_RETURN_IF_ERROR(SetFileContents(out_path, text_proto));
  } else {
    std::cout << text_proto << "\n";
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(argv[0], argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << kUsage;
  }

  return xls::ExitStatus(xls::RealMain());
}
