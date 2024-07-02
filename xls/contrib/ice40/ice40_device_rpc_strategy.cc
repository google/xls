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

#include "xls/contrib/ice40/ice40_device_rpc_strategy.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <ios>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strerror.h"
#include "xls/ir/bit_push_buffer.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

std::string SpeedString(speed_t s) {
  switch (s) {
    case B0:
      return "hang up";
    case B50:
      return "50 baud";
    case B9600:
      return "9600 baud";
    case B19200:
      return "19200 baud";
    case B38400:
      return "38400 baud";
    case B57600:
      return "57600 baud";
    case B115200:
      return "115200 baud";
    default:
      return absl::StrCat("<unhandled speed ", s, ">");
  }
}

// As described in
// https://www.ftdichip.com/Support/Documents/TechnicalNotes/TN_100_USB_VID-PID_Guidelines.pdf
// Section 3.1.
constexpr const char* kVendorId = "0403";
constexpr const char* kProductId = "6010";

// Returns whether the device at symlink_path is in fact an ICE40-looking
// endpoint (as determined by USB vendor/product ID for the FTDI serial-over-USB
// endpoint).
absl::StatusOr<bool> IsDeviceMatch(const std::filesystem::path& symlink_path) {
  std::error_code ec;
  std::filesystem::path real_path =
      std::filesystem::canonical(symlink_path, ec);
  if (ec) {
    return absl::InternalError(absl::StrFormat(
        "Could not resolve symlink path to real path: \"%s\"", symlink_path));
  }
  std::filesystem::path function_path = real_path.parent_path();
  std::filesystem::path device_path = function_path.parent_path();
  XLS_ASSIGN_OR_RETURN(std::string id_vendor_contents,
                       GetFileContents(device_path / "idVendor"));
  XLS_ASSIGN_OR_RETURN(std::string id_product_contents,
                       GetFileContents(device_path / "idProduct"));
  return id_vendor_contents == kVendorId && id_product_contents == kProductId;
}

// Finds the tty device path for the "device_ordinal" ICE40 device attached to
// this host.
absl::StatusOr<std::string> FindPath(int64_t device_ordinal) {
  std::vector<std::string> device_paths;

  XLS_ASSIGN_OR_RETURN(std::vector<std::filesystem::path> usb_device_paths,
                       GetDirectoryEntries("/sys/bus/usb-serial/devices"));
  for (const auto& dir : usb_device_paths) {
    XLS_ASSIGN_OR_RETURN(bool is_device_match, IsDeviceMatch(dir));
    if (is_device_match) {
      // Note: we're assuming a particular udev setup here -- any way to
      // reverse-lookup the dev nodes that have been created via SysFS?
      device_paths.push_back("/dev" / dir.filename());
    }
  }
  if (device_ordinal < device_paths.size()) {
    XLS_RET_CHECK_GE(device_ordinal, 0);
    return device_paths[device_ordinal];
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Device ordinal %d was out of range of %d ICE40 devices found.",
      device_ordinal, device_paths.size()));
}

std::string InputModesToString(tcflag_t input_modes) {
  std::vector<std::string> pieces;
  if (input_modes & IXOFF) {
    pieces.push_back("start/stop input control");
  }
  if (input_modes & IXON) {
    pieces.push_back("start/stop output control");
  }
  return absl::StrJoin(pieces, ", ");
}

std::string ControlModesToString(tcflag_t control_modes) {
  std::vector<std::string> pieces;
  if (int csize = control_modes & CSIZE) {
    if (csize == CS5) {
      pieces.push_back("character size 5b");
    } else if (csize == CS6) {
      pieces.push_back("character size 6b");
    } else if (csize == CS7) {
      pieces.push_back("character size 7b");
    } else if (csize == CS8) {
      pieces.push_back("character size 8b");
    } else {
      pieces.push_back(absl::StrCat("character size value: ", csize));
    }
  }
  if (control_modes & CSTOPB) {
    pieces.push_back("two stop bits");
  } else {
    pieces.push_back("one stop bit");
  }
  if (control_modes & PARENB) {
    pieces.push_back("parity on");
  } else {
    pieces.push_back("parity off");
  }
  return absl::StrJoin(pieces, ", ");
}

std::string LocalModesToString(tcflag_t local_modes) {
  std::vector<std::string> pieces;
  if (local_modes & NOFLSH) {
    pieces.push_back("flush after interrupt-or-quit is off");
  } else {
    pieces.push_back("flush after interrupt-or-quit is on");
  }
  if (local_modes & ICANON) {
    pieces.push_back("canonical input on");
  } else {
    pieces.push_back("canonical input off");
  }
  return absl::StrJoin(pieces, ", ");
}

}  // namespace

Ice40DeviceRpcStrategy::~Ice40DeviceRpcStrategy() {
  if (tty_fd_.has_value()) {
    close(tty_fd_.value());
    tty_fd_ = std::nullopt;
  }
}

absl::Status Ice40DeviceRpcStrategy::Connect(int64_t device_ordinal) {
  if (tty_fd_.has_value()) {
    return absl::FailedPreconditionError(
        "Already connected to an ICE40 device.");
  }
  XLS_ASSIGN_OR_RETURN(std::string path, FindPath(device_ordinal));
  LOG(INFO) << "Found path: " << path
            << " for ICE40 device ordinal: " << device_ordinal;

  // TODO(leary): 2019-04-07 Probably want a way to have the user set or a way
  // to automatically discover the appropriate baud rate for the design
  // currently loaded on the device.
  int fd = open(path.c_str(), O_RDWR);
  if (fd < 0) {
    return absl::InternalError(
        absl::StrFormat("Could not open path as file descriptor: %s; got: %s",
                        path, Strerror(errno)));
  }

  struct termios t;
  if (tcgetattr(fd, &t) != 0) {
    return absl::InternalError(absl::StrFormat(
        "Could not retrieve terminal attributes from file descriptor; got: %s",
        Strerror(errno)));
  }

  // Try to put the terminal into as raw of a mode as we possibly can by turning
  // off lots of stuff -- we're just shuttling bytes over the serial connection,
  // we don't have terminal escapes or anything.
  t.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
  t.c_oflag &= ~OPOST;
  t.c_lflag &= ~(ICANON | ECHO | IEXTEN | ISIG);
  t.c_cc[VINTR] = 0;
  t.c_cc[VQUIT] = 0;
  t.c_cc[VERASE] = 0;
  t.c_cc[VKILL] = 0;
  t.c_cc[VEOF] = 0;
  t.c_cc[VTIME] = 0;
  t.c_cc[VMIN] = 1;  // blocking read until 1 character arrives
#ifdef VSWTC
  t.c_cc[VSWTC] = 0;
#endif
  t.c_cc[VSTART] = 0;
  t.c_cc[VSTOP] = 0;
  t.c_cc[VSUSP] = 0;
  t.c_cc[VEOL] = 0;
  t.c_cc[VREPRINT] = 0;
  t.c_cc[VDISCARD] = 0;
  t.c_cc[VWERASE] = 0;
  t.c_cc[VLNEXT] = 0;
  t.c_cc[VEOL2] = 0;

  if (tcsetattr(fd, TCSANOW, &t) != 0) {
    return absl::InternalError("Could not set terminal to raw mode.");
  }

  if (tcgetattr(fd, &t) != 0) {
    return absl::InternalError(absl::StrFormat(
        "Could not retrieve terminal attributes from file descriptor; got: %s",
        Strerror(errno)));
  }

  speed_t input_speed = cfgetispeed(&t);
  speed_t output_speed = cfgetospeed(&t);

  LOG(INFO) << "input speed:  " << SpeedString(input_speed);
  LOG(INFO) << "output speed: " << SpeedString(output_speed);

  VLOG(1) << "input modes:   " << InputModesToString(t.c_iflag);
  VLOG(1) << "control modes: " << ControlModesToString(t.c_cflag);
  VLOG(1) << "local modes:   " << LocalModesToString(t.c_lflag);

#ifdef LINUX
  VLOG(1) << "c_line: " << std::hex << static_cast<int>(t.c_line);
#endif
  for (int i = 0; i < NCCS; ++i) {
    VLOG(3) << "c_cc[" << i << "]: " << std::hex << static_cast<int>(t.c_cc[i]);
  }

  tty_fd_ = fd;
  return absl::OkStatus();
}

absl::StatusOr<Value> Ice40DeviceRpcStrategy::CallUnnamed(
    const FunctionType& function_type, absl::Span<const Value> arguments) {
  BitPushBuffer buffer;
  for (const Value& arg : arguments) {
    arg.FlattenTo(&buffer);
  }

  if (buffer.empty()) {
    // TODO(leary): 2019-04-07 We probably want this to be possible eventually,
    // but we'd have to decide whether in this case the device function is
    // constantly producing output data since there's no input event to trigger
    // it, so we'd just move on to the read itself.
    return absl::InvalidArgumentError("Cannot perform an empty-payload RPC.");
  }

  std::vector<uint8_t> u8_data = buffer.GetUint8Data();

  int64_t bytes_written = 0;
  while (bytes_written < buffer.size_in_bytes()) {
    int ret = write(tty_fd_.value(), u8_data.data() + bytes_written,
                    u8_data.size() - bytes_written);
    if (ret < 0) {
      return absl::InternalError(
          absl::StrFormat("Could not write partial data of %d remaining bytes "
                          "(originally %d) to ICE40: %s",
                          buffer.size_in_bytes() - bytes_written,
                          buffer.size_in_bytes(), Strerror(errno)));
    }
    bytes_written += ret;
  }

  if (tcflush(tty_fd_.value(), TCOFLUSH) != 0) {
    return absl::InternalError("Could not flush write(s) to device.");
  }

  int64_t output_bits = function_type.return_type()->GetFlatBitCount();
  std::vector<uint8_t> result(CeilOfRatio(output_bits, int64_t{8}));

  VLOG(3) << "Reading device response; expecting " << result.size()
          << " bytes.";

  int64_t bytes_read = 0;
  while (bytes_read < result.size()) {
    int ret = read(tty_fd_.value(), result.data() + bytes_read,
                   result.size() - bytes_read);
    if (ret < 0) {
      return absl::InternalError(
          absl::StrFormat("Could not read partial data of %d remaining bytes "
                          "(originally %d) from ICE40: %s",
                          buffer.size_in_bytes() - bytes_written,
                          buffer.size_in_bytes(), Strerror(errno)));
    }
    bytes_read += ret;
  }

  if (function_type.return_type()->IsBits() &&
      function_type.return_type()->AsBitsOrDie()->bit_count() == 8) {
    return Value(UBits(result[0], 8));
  }

  if (function_type.return_type()->IsBits() &&
      function_type.return_type()->AsBitsOrDie()->bit_count() == 32) {
    return Value(UBits(*absl::bit_cast<uint32_t*>(result.data()), 32));
  }

  return absl::UnimplementedError("NYI: convert result to Value");
}

}  // namespace xls
