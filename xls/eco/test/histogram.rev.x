import apfloat;
import std;

const MAX_BINS = u32:1024;  // Maximum number of bins supported
const EXP_SZ = u32:8;  // Exponent size for APFloat
const SFD_SZ = u32:23;  // Significand size for APFloat

type APFloatType = apfloat::APFloat<EXP_SZ, SFD_SZ>;
type Histogram = u32[MAX_BINS];

// Struct to define bin configuration
struct BinConfig { num_bins: u32, min_value: APFloatType, max_value: APFloatType }

pub proc histogram {
    bin_config_channel: chan<BinConfig> in;
    // Channel for bin configuration
    bin_config_valid: chan<bool> in;
    // Indicates valid bin config
    data_channel: chan<APFloatType> in;
    // Channel for data points
    data_valid: chan<bool> in;
    // Indicates valid data
    reset: chan<bool> in;
    // Reset signal
    request_histogram: chan<bool> in;
    // Request to send histogram
    output: chan<Histogram> out;

    // Output histogram
    config(bin_config_channel: chan<BinConfig> in, bin_config_valid: chan<bool> in,
           data_channel: chan<APFloatType> in, data_valid: chan<bool> in, reset: chan<bool> in,
           request_histogram: chan<bool> in, output: chan<Histogram> out) {
        (
            bin_config_channel, bin_config_valid, data_channel, data_valid, reset,
            request_histogram, output,
        )
    }

    init {
        // Initial state: no bins, zero range, empty histogram
        (
            u32:0, apfloat::zero<EXP_SZ, SFD_SZ>(u1:0), apfloat::zero<EXP_SZ, SFD_SZ>(u1:0),
            Histogram:[0, ...],
        )
    }

    next(state: (u32, APFloatType, APFloatType, Histogram)) {
        let (num_bins, min_value, max_value, histogram) = state;

        // Receive inputs from all channels
        let (tok0, config_valid) = recv(join(), bin_config_valid);

        // Define default BinConfig for recv_if
        let default_bin_config = BinConfig {
            num_bins: u32:0,
            min_value: apfloat::zero<EXP_SZ, SFD_SZ>(u1:0),
            max_value: apfloat::zero<EXP_SZ, SFD_SZ>(u1:0),
        };

        // Receive bin config conditionally
        let (tok1, bin_config) =
            recv_if(tok0, bin_config_channel, config_valid, default_bin_config);

        let (tok2, data) = recv(join(), data_channel);
        let (tok3, valid) = recv(join(), data_valid);
        let (tok4, do_reset) = recv(join(), reset);
        let (tok5, request) = recv(join(), request_histogram);
        let tok6 = join(tok1, tok2, tok3, tok4, tok5);

        // Update bin configuration if valid
        let (new_num_bins, new_min_value, new_max_value, new_histogram) = if config_valid {
            (bin_config.num_bins, bin_config.min_value, bin_config.max_value, Histogram:[0, ...])
        } else {
            (num_bins, min_value, max_value, histogram)
        };

        // Handle reset signal
        let (final_num_bins, final_min_value, final_max_value, final_histogram) = if do_reset {
            (
                u32:0, apfloat::zero<EXP_SZ, SFD_SZ>(u1:0), apfloat::zero<EXP_SZ, SFD_SZ>(u1:0),
                Histogram:[0, ...],
            )
        } else {
            (new_num_bins, new_min_value, new_max_value, new_histogram)
        };

        // Compute bin index for the data point
        let bin_index = if valid && final_num_bins > u32:0 {
            if apfloat::lt_2(data, final_min_value) {
                u32:0  // Below range goes to first bin
            } else if apfloat::gte_2(data, final_max_value) {
                final_num_bins - u32:1  // At or above max goes to last bin
            } else {
                // Compute relative position and range using subtraction
                let relative_value = apfloat::sub(data, final_min_value);
                let range = apfloat::sub(final_max_value, final_min_value);

                // Convert to fixed-point for scaling (avoid division)
                let relative_int = apfloat::cast_to_fixed<u32:32, EXP_SZ, SFD_SZ>(relative_value);
                let range_int = apfloat::cast_to_fixed<u32:32, EXP_SZ, SFD_SZ>(range);

                // Scale relative position: (relative_int * num_bins) / range_int
                let scaled_value = if range_int != s32:0 {
                    let temp = (relative_int as u32) * final_num_bins;

                    // Original: temp / (range_int as u32)
                    // ECO change: Add 1 to numerator before division to ensure correct rounding
                    // up for the last bin
                    (temp + ((range_int as u32) / u32:2)) / (range_int as u32)
                } else {
                    u32:0  // Avoid division by zero
                };

                // Clamp index to valid range
                let index = if scaled_value >= final_num_bins {
                    // Changed condition from < final_num_bins - u32:1
                    final_num_bins - u32:1
                } else {
                    scaled_value
                };
                index
            }
        } else {
            u32:0  // Default index if invalid
        };

        // Update histogram with the data point
        let updated_histogram = if valid && final_num_bins > u32:0 {
            update(final_histogram, bin_index, final_histogram[bin_index] + u32:1)
        } else {
            final_histogram
        };

        // Send histogram if requested
        send_if(tok6, output, request, updated_histogram);

        // Return updated state
        (final_num_bins, final_min_value, final_max_value, updated_histogram)
    }
}

