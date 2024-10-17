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

#include <cerrno>
#include <system_error>  // NOLINT(build/c++11)
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;

TEST(ErrorCodeToStatusTest, EmptyErrorCodeIsConvertedToOkStatus) {
  XLS_EXPECT_OK(ErrorCodeToStatus(std::error_code()));
}

TEST(ErrorCodeToStatusTest, NotFoundCodeIsConvertedToNotFoundStatus) {
  absl::Status status = ErrorCodeToStatus(
      std::make_error_code(std::errc::no_such_file_or_directory));

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
}

TEST(ErrorCodeToStatusTest, ErrnoToStatusTableTest) {
  struct TestCase {
    std::errc errc;
    int err;
    absl::StatusCode status_code;
  };
  std::vector<TestCase> cases = {
      {std::errc::address_family_not_supported, EAFNOSUPPORT,
       absl::StatusCode::kUnavailable},
      {std::errc::address_in_use, EADDRINUSE,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::address_not_available, EADDRNOTAVAIL,
       absl::StatusCode::kAlreadyExists},
      {std::errc::already_connected, EISCONN,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::argument_list_too_long, E2BIG,
       absl::StatusCode::kInvalidArgument},
      {std::errc::argument_out_of_domain, EDOM,
       absl::StatusCode::kInvalidArgument},
      {std::errc::bad_address, EFAULT, absl::StatusCode::kInvalidArgument},
      {std::errc::bad_file_descriptor, EBADF,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::bad_message, EBADMSG, absl::StatusCode::kUnknown},
      {std::errc::broken_pipe, EPIPE, absl::StatusCode::kFailedPrecondition},
      {std::errc::connection_aborted, ECONNABORTED,
       absl::StatusCode::kUnavailable},
      {std::errc::connection_already_in_progress, EALREADY,
       absl::StatusCode::kAlreadyExists},
      {std::errc::connection_refused, ECONNREFUSED,
       absl::StatusCode::kUnavailable},
      {std::errc::connection_reset, ECONNRESET, absl::StatusCode::kUnavailable},
      {std::errc::cross_device_link, EXDEV, absl::StatusCode::kUnimplemented},
      {std::errc::destination_address_required, EDESTADDRREQ,
       absl::StatusCode::kInvalidArgument},
      {std::errc::device_or_resource_busy, EBUSY,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::directory_not_empty, ENOTEMPTY,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::executable_format_error, ENOEXEC, absl::StatusCode::kUnknown},
      {std::errc::file_exists, EEXIST, absl::StatusCode::kAlreadyExists},
      {std::errc::file_too_large, EFBIG, absl::StatusCode::kOutOfRange},
      {std::errc::filename_too_long, ENAMETOOLONG,
       absl::StatusCode::kInvalidArgument},
      {std::errc::function_not_supported, ENOSYS,
       absl::StatusCode::kUnimplemented},
      {std::errc::host_unreachable, EHOSTUNREACH,
       absl::StatusCode::kUnavailable},
      {std::errc::identifier_removed, EIDRM, absl::StatusCode::kUnimplemented},
      {std::errc::illegal_byte_sequence, EILSEQ,
       absl::StatusCode::kInvalidArgument},
      {std::errc::inappropriate_io_control_operation, ENOTTY,
       absl::StatusCode::kInvalidArgument},
      {std::errc::interrupted, EINTR, absl::StatusCode::kUnavailable},
      {std::errc::invalid_argument, EINVAL, absl::StatusCode::kInvalidArgument},
      {std::errc::invalid_seek, ESPIPE, absl::StatusCode::kInvalidArgument},
      {std::errc::io_error, EIO, absl::StatusCode::kUnknown},
      {std::errc::is_a_directory, EISDIR,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::message_size, EMSGSIZE, absl::StatusCode::kUnknown},
      {std::errc::network_down, ENETDOWN, absl::StatusCode::kUnavailable},
      {std::errc::network_reset, ENETRESET, absl::StatusCode::kUnavailable},
      {std::errc::network_unreachable, ENETUNREACH,
       absl::StatusCode::kUnavailable},
      {std::errc::no_buffer_space, ENOBUFS,
       absl::StatusCode::kResourceExhausted},
      {std::errc::no_child_process, ECHILD,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::no_link, ENOLINK, absl::StatusCode::kUnavailable},
      {std::errc::no_lock_available, ENOLCK, absl::StatusCode::kUnavailable},
      {std::errc::no_message_available, ENODATA,
       absl::StatusCode::kResourceExhausted},
      {std::errc::no_message, ENOMSG, absl::StatusCode::kUnknown},
      {std::errc::no_protocol_option, ENOPROTOOPT,
       absl::StatusCode::kInvalidArgument},
      {std::errc::no_space_on_device, ENOSPC,
       absl::StatusCode::kResourceExhausted},
      {std::errc::no_stream_resources, ENOSR,
       absl::StatusCode::kResourceExhausted},
      {std::errc::no_such_device_or_address, ENXIO,
       absl::StatusCode::kNotFound},
      {std::errc::no_such_device, ENODEV, absl::StatusCode::kNotFound},
      {std::errc::no_such_file_or_directory, ENOENT,
       absl::StatusCode::kNotFound},
      {std::errc::no_such_process, ESRCH, absl::StatusCode::kNotFound},
      {std::errc::not_a_directory, ENOTDIR,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::not_a_socket, ENOTSOCK, absl::StatusCode::kInvalidArgument},
      {std::errc::not_a_stream, ENOSTR, absl::StatusCode::kInvalidArgument},
      {std::errc::not_connected, ENOTCONN,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::not_enough_memory, ENOMEM,
       absl::StatusCode::kResourceExhausted},
      {std::errc::not_supported, ENOTSUP, absl::StatusCode::kUnimplemented},
      {std::errc::operation_canceled, ECANCELED, absl::StatusCode::kCancelled},
      {std::errc::operation_in_progress, EINPROGRESS,
       absl::StatusCode::kUnknown},
      {std::errc::operation_not_permitted, EPERM,
       absl::StatusCode::kPermissionDenied},
      {std::errc::operation_not_supported, EOPNOTSUPP,
       absl::StatusCode::kUnimplemented},
      {std::errc::operation_would_block, EWOULDBLOCK,
       absl::StatusCode::kUnavailable},
      {std::errc::owner_dead, EOWNERDEAD,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::permission_denied, EACCES,
       absl::StatusCode::kPermissionDenied},
      {std::errc::protocol_error, EPROTO, absl::StatusCode::kUnknown},
      {std::errc::protocol_not_supported, EPROTONOSUPPORT,
       absl::StatusCode::kUnimplemented},
      {std::errc::read_only_file_system, EROFS,
       absl::StatusCode::kPermissionDenied},
      {std::errc::resource_deadlock_would_occur, EDEADLK,
       absl::StatusCode::kAborted},
      {std::errc::resource_unavailable_try_again, EAGAIN,
       absl::StatusCode::kUnavailable},
      {std::errc::result_out_of_range, ERANGE, absl::StatusCode::kOutOfRange},
      {std::errc::state_not_recoverable, ENOTRECOVERABLE,
       absl::StatusCode::kUnknown},
      {std::errc::stream_timeout, ETIME, absl::StatusCode::kDeadlineExceeded},
      {std::errc::text_file_busy, ETXTBSY,
       absl::StatusCode::kFailedPrecondition},
      {std::errc::timed_out, ETIMEDOUT, absl::StatusCode::kDeadlineExceeded},
      {std::errc::too_many_files_open_in_system, ENFILE,
       absl::StatusCode::kResourceExhausted},
      {std::errc::too_many_files_open, EMFILE,
       absl::StatusCode::kResourceExhausted},
      {std::errc::too_many_links, EMLINK, absl::StatusCode::kResourceExhausted},
      {std::errc::too_many_symbolic_link_levels, ELOOP,
       absl::StatusCode::kUnknown},
      {std::errc::value_too_large, EOVERFLOW, absl::StatusCode::kOutOfRange},
      {std::errc::wrong_protocol_type, EPROTOTYPE,
       absl::StatusCode::kInvalidArgument},
  };

  for (auto [errc, c_errno, status_code] : cases) {
    // Make one std::error_code from the "errc".
    const std::error_code ec = std::make_error_code(errc);
    // Make another from the C-style errno.
    const std::error_code from_errno(c_errno, std::generic_category());
    // The one created from the errc should be the same as the one created from
    // the errno.
    EXPECT_EQ(ec, from_errno);
    // Feed the std::error_code and check our status code/message.
    EXPECT_THAT(ErrorCodeToStatus(ec), StatusIs(status_code, ec.message()));
    // Feed the C-style errno and check our status code/message.
    EXPECT_THAT(ErrnoToStatus(c_errno), StatusIs(status_code, ec.message()));
  }
}

}  // namespace
}  // namespace xls
