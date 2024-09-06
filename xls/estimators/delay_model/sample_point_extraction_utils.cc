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

#include "xls/estimators/delay_model/sample_point_extraction_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/case_converters.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/estimator_model.pb.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls::estimator_model {
namespace {

using SampleMap =
    absl::btree_map<std::string, absl::btree_map<std::string, SamplePoint>>;

constexpr char kIdentityOpName[] = "kIdentity";

// A categorization of XLS IR operations to inform how we should extract samples
// for them. Ops with a regression estimator need delay estimates. Ops that have
// a key in `aliases` are aliases to the op that is the corresponding value, and
// their samples should be attributed to the value op.
struct ExtractionConfig {
  absl::flat_hash_set<std::string> ops_with_regression_estimator;
  absl::flat_hash_map<std::string, std::string> aliases;
};

// Given an `OpModels` proto, creates an `ExtractionConfig` that puts relevant
// ops in the two categorizes we care about.
ExtractionConfig CategorizeOps(const OpModels& op_models) {
  absl::flat_hash_set<std::string> ops;
  ExtractionConfig config;
  for (const auto& model : op_models.op_models()) {
    if (model.estimator().has_regression()) {
      config.ops_with_regression_estimator.insert(model.op());
    }
    if (model.estimator().has_alias_op()) {
      config.aliases.emplace(model.op(), model.estimator().alias_op());
    }
  }
  return config;
}

// Returns the attributes that need to be on an `OpSamples` message with the
// given `op_string`, in order for the timing characterization client to do the
// right thing downstream. For normal delay models, these are handwritten in the
// input samples file. Most ops do not need any attributes.
std::optional<std::string_view> GetOpAttributes(std::string_view op_string) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kAttributes(absl::flat_hash_map<std::string, std::string>{
          {"kSignExt", "new_bit_count=%r"},
          {"kDynamicBitSlice", "width=%r"},
          {"kOneHot", "lsb_prio=true"}});
  const auto it = kAttributes->find(op_string);
  if (it == kAttributes->end()) {
    return std::nullopt;
  }
  return it->second;
}

// Creates a key for the given parameterization that is usable for both mapping
// and sorting in a canonical order.
std::string CreateKey(const Parameterization& sample) {
  std::string out;
  absl::StrAppendFormat(&out, "result width: %08d;", sample.result_width());
  for (int64_t width : sample.operand_widths()) {
    absl::StrAppendFormat(&out, " operand width: %08d;", width);
  }
  return out;
}

// Returns the op name that `op_name` is an alias of. If it is not an alias,
// then returns `op_name`.
std::string_view ResolveOpNameIfAlias(const ExtractionConfig& config,
                                      std::string_view op_name) {
  const auto it = config.aliases.find(op_name);
  return it == config.aliases.end() ? op_name : it->second;
}

// Given an `array` that may be a multi-dimensional container of other arrays,
// this function returns all the dimensions, from outermost to innermost.
std::vector<int64_t> UnwrapArrayDims(const xls::ArrayType* array) {
  std::vector<int64_t> dims;
  while (array) {
    dims.push_back(array->size());
    array = array->element_type()->IsArray()
                ? array->element_type()->AsArrayOrDie()
                : nullptr;
  }
  return dims;
}

// Adds a `kIdentity` sample to the given `list`. This is a synthetic
// operation used to gauge the nonzero delay of a design that "does nothing", to
// use as an offset for the delays of real operations.
void AddIdentitySample(OpSamplesList& list) {
  OpSamples* identity = list.add_op_samples();
  identity->set_op(kIdentityOpName);
  Parameterization* identity_params = identity->add_samples();
  identity_params->set_result_width(1);
  identity_params->add_operand_widths(1);
}

// Creates a `SamplePoint` for the given node, if it is a type of node that
// should get a delay estimate.
std::optional<SamplePoint> MaybeCreateSamplePoint(
    const ExtractionConfig& config, const Node* node) {
  const std::string op_name(ResolveOpNameIfAlias(
      config, absl::StrCat("k", Camelize(OpToString(node->op())))));
  if (!config.ops_with_regression_estimator.contains(op_name)) {
    return std::nullopt;
  }
  SamplePoint point{.op_name = op_name};
  if (node->GetType()->IsBits()) {
    point.params.set_result_width(node->BitCountOrDie());
  } else if (node->GetType()->IsArray()) {
    for (int64_t dim : UnwrapArrayDims(node->GetType()->AsArrayOrDie())) {
      point.params.add_result_element_counts(dim);
    }
  } else {
    point.params.set_result_width(0);
  }
  int operand_number = 0;
  for (Node* operand : node->operands()) {
    if (operand->GetType()->IsBits()) {
      point.params.add_operand_widths(operand->BitCountOrDie());
    } else if (operand->GetType()->IsArray()) {
      xls::ArrayType* array = operand->GetType()->AsArrayOrDie();
      point.params.add_operand_widths(array->element_type()->GetFlatBitCount());
      OperandElementCounts* counts = point.params.add_operand_element_counts();
      counts->set_operand_number(operand_number);
      for (int64_t dim : UnwrapArrayDims(array)) {
        counts->add_element_counts(dim);
      }
    } else {
      point.params.add_operand_widths(0);
    }
    operand_number++;
  }
  return point;
}

// Flattens the given `samples` map into a vector in canonical order.
std::vector<SamplePoint> ConvertSampleMapToVector(SampleMap samples) {
  std::vector<SamplePoint> points;
  for (auto& [_, next_points] : samples) {
    for (auto& [_, point] : next_points) {
      points.push_back(std::move(point));
    }
  }
  return points;
}

// Flattens the given `samples` map into a proto in canonical order.
OpSamplesList ConvertSampleMapToList(SampleMap samples) {
  OpSamplesList result;
  AddIdentitySample(result);
  int count = 0;
  for (auto& [op_string, samples_for_op] : samples) {
    OpSamples* samples_proto = result.add_op_samples();
    samples_proto->set_op(op_string);
    std::optional<std::string_view> attributes = GetOpAttributes(op_string);
    if (attributes.has_value()) {
      samples_proto->set_attributes(*attributes);
    }
    for (auto& [_, sample] : samples_for_op) {
      samples_proto->add_samples()->Swap(&sample.params);
      ++count;
    }
  }
  VLOG(2) << "Processed " << count << " samples.";
  return result;
}

}  // namespace

absl::StatusOr<std::vector<SamplePoint>> ExtractSamplePoints(
    const Package& package, const estimator_model::OpModels& op_models,
    std::optional<DelayEstimator*> delay_estimator) {
  ExtractionConfig config = CategorizeOps(op_models);
  SampleMap samples;
  for (FunctionBase* fn : package.GetFunctionBases()) {
    for (Node* node : fn->nodes()) {
      std::optional<SamplePoint> point = MaybeCreateSamplePoint(config, node);
      if (point.has_value()) {
        SamplePoint& mapped_point =
            samples[point->op_name]
                .emplace(CreateKey(point->params), std::move(*point))
                .first->second;
        if (mapped_point.frequency == 0 && delay_estimator.has_value()) {
          XLS_ASSIGN_OR_RETURN(mapped_point.delay_estimate_in_ps,
                               (*delay_estimator)->GetOperationDelayInPs(node));
        }
        ++mapped_point.frequency;
      }
    }
  }
  return ConvertSampleMapToVector(std::move(samples));
}

OpSamplesList ConvertToOpSamplesList(absl::Span<const SamplePoint> samples,
                                     size_t n) {
  SampleMap map;
  size_t count = 0;
  for (const SamplePoint& point : samples) {
    if (point.op_name == kIdentityOpName) {
      // The conversion to a list below will automatically add this.
      continue;
    }
    map[point.op_name].emplace(CreateKey(point.params), point);
    if (++count == n) {
      break;
    }
  }
  return ConvertSampleMapToList(std::move(map));
}

}  // namespace xls::estimator_model
