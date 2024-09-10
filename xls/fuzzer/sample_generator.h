// Copyright 2021 The XLS Authors
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

#ifndef XLS_FUZZER_SAMPLE_GENERATOR_H_
#define XLS_FUZZER_SAMPLE_GENERATOR_H_

#include "absl/random/bit_gen_ref.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample.h"

namespace xls {

// Generates and returns a random Sample with the given options.
absl::StatusOr<Sample> GenerateSample(
    const dslx::AstGeneratorOptions& generator_options,
    const SampleOptions& sample_options, absl::BitGenRef bit_gen,
    dslx::FileTable& file_table);

}  // namespace xls

#endif  // XLS_FUZZER_SAMPLE_GENERATOR_H_
