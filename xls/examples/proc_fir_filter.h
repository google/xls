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
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_EXAMPLES_PROC_FIR_FILTER_H_
#define XLS_EXAMPLES_PROC_FIR_FILTER_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {

// Defines a finite impulse response (FIR) filter as a proc.
// FIR filters are commonly used for Digital Signal Processing (DSP)
// applications as they implement a convolution.
// There are many use cases of FIR filters as the concept of convolution is very
// general. For example, a convolution in time domain is equivalent
// to multiplication in the Fourier domain, so a frequent use of an FIR filter
// is to implement a low or high pass filter. A Sobel edge detector in computer
// graphics is an FIR filter. In the wireless world, FIR filters can be used to
// model and therefore mitigate channel effects, which is an important step in
// decoding a received signal in certain cases.
absl::StatusOr<Proc*> CreateFirFilter(std::string_view name,
                                      const Value& kernel,
                                      StreamingChannel* input_channel,
                                      StreamingChannel* output_channel,
                                      Package* package);

}  // namespace xls

#endif  // XLS_EXAMPLES_PROC_FIR_FILTER_H_
