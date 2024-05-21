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

#include "xls/common/status/error_code_to_status.h"

#include <system_error>  // NOLINT(build/c++11)

#include "absl/status/status.h"
#include "xls/common/status/status_builder.h"

namespace xls {

absl::StatusCode ErrorCodeToStatusCode(const std::error_code& ec) {
  if (!ec) {
    return absl::StatusCode::kOk;
  }
  if (ec.category() != std::generic_category()) {
    return absl::StatusCode::kUnknown;
  }

  std::errc ec_value = static_cast<std::errc>(ec.value());

  // Note: EOPNOTSUPP and ENOTSUP have the same value on Linux, but are
  // different on BSD derivatives. The compiler complains about duplicate case
  // values on Linux, so handle them specially outside the switch.
  if (ec_value == std::errc::operation_not_supported ||  // EOPNOTSUPP
      ec_value == std::errc::not_supported) {            // ENOTSUP
    return absl::StatusCode::kUnimplemented;
  }

  switch (ec_value) {
    case std::errc::address_family_not_supported:  // EAFNOSUPPORT
      return absl::StatusCode::kUnavailable;
    case std::errc::address_in_use:  // EADDRINUSE
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::address_not_available:  // EADDRNOTAVAIL
      return absl::StatusCode::kAlreadyExists;
    case std::errc::already_connected:  // EISCONN
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::argument_list_too_long:  // E2BIG
      return absl::StatusCode::kInvalidArgument;
    case std::errc::argument_out_of_domain:  // EDOM
      return absl::StatusCode::kInvalidArgument;
    case std::errc::bad_address:  // EFAULT
      return absl::StatusCode::kInvalidArgument;
    case std::errc::bad_file_descriptor:  // EBADF
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::bad_message:  // EBADMSG
      return absl::StatusCode::kUnknown;
    case std::errc::broken_pipe:  // EPIPE
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::connection_aborted:  // ECONNABORTED
      return absl::StatusCode::kUnavailable;
    case std::errc::connection_already_in_progress:  // EALREADY
      return absl::StatusCode::kAlreadyExists;
    case std::errc::connection_refused:  // ECONNREFUSED
      return absl::StatusCode::kUnavailable;
    case std::errc::connection_reset:  // ECONNRESET
      return absl::StatusCode::kUnavailable;
    case std::errc::cross_device_link:  // EXDEV
      return absl::StatusCode::kUnimplemented;
    case std::errc::destination_address_required:  // EDESTADDRREQ
      return absl::StatusCode::kInvalidArgument;
    case std::errc::device_or_resource_busy:  // EBUSY
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::directory_not_empty:  // ENOTEMPTY
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::executable_format_error:  // ENOEXEC
      return absl::StatusCode::kUnknown;
    case std::errc::file_exists:  // EEXIST
      return absl::StatusCode::kAlreadyExists;
    case std::errc::file_too_large:  // EFBIG
      return absl::StatusCode::kOutOfRange;
    case std::errc::filename_too_long:  // ENAMETOOLONG
      return absl::StatusCode::kInvalidArgument;
    case std::errc::function_not_supported:  // ENOSYS
      return absl::StatusCode::kUnimplemented;
    case std::errc::host_unreachable:  // EHOSTUNREACH
      return absl::StatusCode::kUnavailable;
    case std::errc::identifier_removed:  // EIDRM
      return absl::StatusCode::kUnimplemented;
    case std::errc::illegal_byte_sequence:  // EILSEQ
      return absl::StatusCode::kInvalidArgument;
    case std::errc::inappropriate_io_control_operation:  // ENOTTY
      return absl::StatusCode::kInvalidArgument;
    case std::errc::interrupted:  // EINTR
      return absl::StatusCode::kUnavailable;
    case std::errc::invalid_argument:  // EINVAL
      return absl::StatusCode::kInvalidArgument;
    case std::errc::invalid_seek:  // ESPIPE
      return absl::StatusCode::kInvalidArgument;
    case std::errc::io_error:  // EIO
      return absl::StatusCode::kUnknown;
    case std::errc::is_a_directory:  // EISDIR
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::message_size:  // EMSGSIZE
      return absl::StatusCode::kUnknown;
    case std::errc::network_down:  // ENETDOWN
      return absl::StatusCode::kUnavailable;
    case std::errc::network_reset:  // ENETRESET
      return absl::StatusCode::kUnavailable;
    case std::errc::network_unreachable:  // ENETUNREACH
      return absl::StatusCode::kUnavailable;
    case std::errc::no_buffer_space:  // ENOBUFS
      return absl::StatusCode::kResourceExhausted;
    case std::errc::no_child_process:  // ECHILD
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::no_link:  // ENOLINK
      return absl::StatusCode::kUnavailable;
    case std::errc::no_lock_available:  // ENOLCK
      return absl::StatusCode::kUnavailable;
    case std::errc::no_message_available:  // ENODATA
      return absl::StatusCode::kResourceExhausted;
    case std::errc::no_message:  // ENOMSG
      return absl::StatusCode::kUnknown;
    case std::errc::no_protocol_option:  // ENOPROTOOPT
      return absl::StatusCode::kInvalidArgument;
    case std::errc::no_space_on_device:  // ENOSPC
      return absl::StatusCode::kResourceExhausted;
    case std::errc::no_stream_resources:  // ENOSR
      return absl::StatusCode::kResourceExhausted;
    case std::errc::no_such_device_or_address:  // ENXIO
      return absl::StatusCode::kNotFound;
    case std::errc::no_such_device:  // ENODEV
      return absl::StatusCode::kNotFound;
    case std::errc::no_such_file_or_directory:  // ENOENT
      return absl::StatusCode::kNotFound;
    case std::errc::no_such_process:  // ESRCH
      return absl::StatusCode::kNotFound;
    case std::errc::not_a_directory:  // ENOTDIR
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::not_a_socket:  // ENOTSOCK
      return absl::StatusCode::kInvalidArgument;
    case std::errc::not_a_stream:  // ENOSTR
      return absl::StatusCode::kInvalidArgument;
    case std::errc::not_connected:  // ENOTCONN
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::not_enough_memory:  // ENOMEM
      return absl::StatusCode::kResourceExhausted;
    case std::errc::operation_canceled:  // ECANCELED
      return absl::StatusCode::kCancelled;
    case std::errc::operation_in_progress:  // EINPROGRESS
      return absl::StatusCode::kUnknown;
    case std::errc::operation_not_permitted:  // EPERM
      return absl::StatusCode::kPermissionDenied;
    case std::errc::owner_dead:  // EOWNERDEAD
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::permission_denied:  // EACCES
      return absl::StatusCode::kPermissionDenied;
    case std::errc::protocol_error:  // EPROTO
      return absl::StatusCode::kUnknown;
    case std::errc::protocol_not_supported:  // EPROTONOSUPPORT
      return absl::StatusCode::kUnimplemented;
    case std::errc::read_only_file_system:  // EROFS
      return absl::StatusCode::kPermissionDenied;
    case std::errc::resource_deadlock_would_occur:  // EDEADLK
      return absl::StatusCode::kAborted;
    // Note: this case has the same value as resource_unavailable_try_again.
    // case std::errc::operation_would_block:  // EWOULDBLOCK
    case std::errc::resource_unavailable_try_again:  // EAGAIN
      return absl::StatusCode::kUnavailable;
    case std::errc::result_out_of_range:  // ERANGE
      return absl::StatusCode::kOutOfRange;
    case std::errc::state_not_recoverable:  // ENOTRECOVERABLE
      return absl::StatusCode::kUnknown;
    case std::errc::stream_timeout:  // ETIME
      return absl::StatusCode::kDeadlineExceeded;
    case std::errc::text_file_busy:  // ETXTBSY
      return absl::StatusCode::kFailedPrecondition;
    case std::errc::timed_out:  // ETIMEDOUT
      return absl::StatusCode::kDeadlineExceeded;
    case std::errc::too_many_files_open_in_system:  // ENFILE
      return absl::StatusCode::kResourceExhausted;
    case std::errc::too_many_files_open:  // EMFILE
      return absl::StatusCode::kResourceExhausted;
    case std::errc::too_many_links:  // EMLINK
      return absl::StatusCode::kResourceExhausted;
    case std::errc::too_many_symbolic_link_levels:  // ELOOP
      return absl::StatusCode::kUnknown;
    case std::errc::value_too_large:  // EOVERFLOW
      return absl::StatusCode::kOutOfRange;
    case std::errc::wrong_protocol_type:  // EPROTOTYPE
      return absl::StatusCode::kInvalidArgument;
    default:
      return absl::StatusCode::kUnknown;
  }
}

xabsl::StatusBuilder ErrorCodeToStatus(const std::error_code& ec) {
  return xabsl::StatusBuilder(
      absl::Status(ErrorCodeToStatusCode(ec), ec.message()));
}

xabsl::StatusBuilder ErrnoToStatus(int errno_value) {
  return xabsl::StatusBuilder(
      ErrorCodeToStatus(std::error_code(errno_value, std::generic_category())));
}

}  // namespace xls
