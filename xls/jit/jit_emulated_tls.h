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

#ifndef XLS_JIT_JIT_EMULATED_TLS_H_
#define XLS_JIT_JIT_EMULATED_TLS_H_

#include <cstdint>
#include <string_view>

// Unmangled c-export symbol that the JIT code will link against.
extern "C" void* c_export_xls_GetEmulatedMsanTLSAddr(void* selector);

namespace xls {

inline constexpr std::string_view kExportedEmulatedMsanEntrypointName =
    "c_export_xls_GetEmulatedMsanTLSAddr";

// Parameter used to call __emutls_get_address for param_tls
inline constexpr uintptr_t kParamTlsEntry = 1;
// Parameter used to call __emutls_get_address for retval_tls
inline constexpr uintptr_t kRetvalTlsEntry = 2;

// Implementation of emulated TLS for jit use. Defines an unmangled symbol for
// jit.
void* GetEmulatedMsanTLSAddr(void* selector);

}  // namespace xls

#endif  // XLS_JIT_JIT_EMULATED_TLS_H_
