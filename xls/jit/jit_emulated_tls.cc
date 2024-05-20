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

#include "xls/jit/jit_emulated_tls.h"

#include <bit>
#include <cstdint>

#include "absl/log/log.h"

extern "C" void* c_export_xls_GetEmulatedMsanTLSAddr(void* selector) {
  return xls::GetEmulatedMsanTLSAddr(selector);
}

namespace xls {

namespace {

#ifdef ABSL_HAVE_MEMORY_SANITIZER
static constexpr bool kHasMsan = true;
// TODO(allight): Technically if we want to support origin-tracking we could but
// we'd need to add more of the locals from MSan.cpp here.
extern "C" __thread unsigned long long  // NOLINT(runtime/int)
    __msan_param_tls[];
extern "C" __thread unsigned long long  // NOLINT(runtime/int)
    __msan_retval_tls[];

#define param_tls __msan_param_tls
#define retval_tls __msan_retval_tls

#else
static constexpr bool kHasMsan = false;
// Some fake definitions for convenience.
static void* param_tls = nullptr;
static void* retval_tls = nullptr;
#endif
}  // namespace

// Based on https://github.com/google/sanitizers/wiki/MemorySanitizerJIT
// tutorial.
// Identifiers we use to pick out the actual thread-local buffers shared with
// host msan. Basically the msan ABI is:
// %x = load-symbol __emutls_v.__msan_param_tls
// %y = load-symbol __emutls_get_address
// %tls_slot = invoke %y (%x)
void* GetEmulatedMsanTLSAddr(void* selector) {
  if (!kHasMsan) {
    LOG(ERROR) << "Unexpected MSAN call on non-msan build?";
    return nullptr;
  }
  VLOG(2) << "EMU_MSAN (enabled: " << kHasMsan << ") Called with "
          << std::bit_cast<uintptr_t>(selector);
  switch (std::bit_cast<uintptr_t>(selector)) {
    case kParamTlsEntry:
      return std::bit_cast<void*>(&param_tls);
    case kRetvalTlsEntry:
      return std::bit_cast<void*>(&retval_tls);
    default:
      LOG(ERROR) << "Unexpected TLS addr request: " << selector;
      return nullptr;
  }
}

}  // namespace xls
