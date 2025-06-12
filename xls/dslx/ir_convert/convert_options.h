// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_IR_CONVERT_CONVERT_OPTIONS_H_
#define XLS_DSLX_IR_CONVERT_CONVERT_OPTIONS_H_

#include <optional>

#include "xls/dslx/warning_kind.h"
#include "xls/ir/channel.h"

namespace xls::dslx {

// Bundles together options (common among the API routines below) used in
// DSLX-to-IR conversion.
struct ConvertOptions {
  // Whether to emit positional metadata into the output IR.
  //
  // Stripping positions can be useful for less fragile string matching in
  // development, e.g. tests.
  bool emit_positions = true;

  // Whether to emit fail!() operations as predicated assertion IR nodes.
  bool emit_fail_as_assert = true;

  // Should the generated IR be verified?
  bool verify_ir = true;

  // Should warnings be treated as errors?
  bool warnings_as_errors = true;

  // Set of warnings used in DSLX typechecking.
  //
  // Note that this is only used in IR conversion routines that do typechecking.
  WarningKindSet warnings = kDefaultWarningsSet;

  // Should #[test] and #[test_proc] entities be emitted to IR.
  bool convert_tests = false;

  // If present, the default FIFO config to use for any FIFO that does not
  // specify a config.
  std::optional<FifoConfig> default_fifo_config;

  // Should we convert to proc-scoped channels after IR conversion with global
  // channels?
  // TODO: https://github.com/google/xls/issues/2078 - Remove this after
  // lower_to_proc_scoped_channels is turned on for all tests.
  bool proc_scoped_channels = false;

  // Whether to use type system v2 to perform type checking.
  bool type_inference_v2 = false;

  // Should we generate proc-scoped channels without global channels as an
  // intermediate step? See https://github.com/google/xls/issues/2078
  bool lower_to_proc_scoped_channels = false;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_CONVERT_OPTIONS_H_
