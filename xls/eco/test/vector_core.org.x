const N = u32:4;           // Length of each vector
const M = u32:4;           // Number of vectors to compare
const COUNT_OFF_DIAGONAL = (M * M) - M;
const PAIR_COUNT = (M * (M - u32:1)) / u32:2;  // For averaging over distinct pairs

// ------------------------------------------------------------
// Basic vector operations
// ------------------------------------------------------------

// Squared Euclidean distance between two vectors.
fn squared_distance(a: u8[N], b: u8[N]) -> u32 {
  for (i, acc) in u32:0..N {
    let diff = if a[i] > b[i] { a[i] - b[i] } else { b[i] - a[i] };
    let diff_squared = (diff as u16) * (diff as u16);
    acc + (diff_squared as u32)
  }(u32:0)  // Removed offset for standard calculation
}

// Manhattan (L1) distance between two vectors.
fn manhattan_distance(a: u8[N], b: u8[N]) -> u32 {
  for (i, acc) in u32:0..N {
    let diff = if a[i] > b[i] { a[i] - b[i] } else { b[i] - a[i] };
    acc + (diff as u32)
  }(u32:0)
}

// Squared magnitude of a vector (renamed for clarity).
fn squared_magnitude(a: u8[N]) -> u32 {
  for (i, acc) in u32:0..N {
    let elem_squared = (a[i] as u16) * (a[i] as u16);
    acc + (elem_squared as u32)
  }(u32:0)
}

// Dot product of two vectors.
fn dot_product(a: u8[N], b: u8[N]) -> u32 {
  for (i, acc) in u32:0..N {
    acc + (a[i] as u32) * (b[i] as u32)
  }(u32:0)
}

// Element-wise vector addition.
fn vector_add(a: u8[N], b: u8[N]) -> u8[N] {
  let initial_result = u8[N]:[0, ...];
  for (i, result): (u32, u8[N]) in u32:0..N {
    update(result, i, a[i] + b[i])
  }(initial_result)
}

// Element-wise vector subtraction.
fn vector_subtract(a: u8[N], b: u8[N]) -> u8[N] {
  let initial_result = u8[N]:[0, ...];
  for (i, result): (u32, u8[N]) in u32:0..N {
    update(result, i, a[i] - b[i])
  }(initial_result)
}

// Scale a vector by a scalar.
fn vector_scale(a: u8[N], scalar: u8) -> u8[N] {
  let initial_result = u8[N]:[0, ...];
  for (i, result): (u32, u8[N]) in u32:0..N {
    update(result, i, (a[i] * scalar) as u8)
  }(initial_result)
}

// Compute the component-wise average of an array of vectors.
fn average_vector(vectors: u8[N][M]) -> u8[N] {
  let sum = for (i, acc): (u32, u32[N]) in u32:0..M {
    let new_acc = for (j, r): (u32, u32[N]) in u32:0..N {
      update(r, j, acc[j] + (vectors[i][j] as u32))
    }(acc);
    new_acc
  }(u32[N]:[0, ...]);
  let initial_avg = u8[N]:[0, ...];
  for (i, result): (u32, u8[N]) in u32:0..N {
    update(result, i, (sum[i] / M) as u8)
  }(initial_avg)
}

// ------------------------------------------------------------
// Additional Analysis Functions
// ------------------------------------------------------------

// Find the nearest vector (using squared distance).
fn find_nearest(refrence: u8[N], vectors: u8[N][M]) -> (u32, u32) {
  for (i, (min_dist, min_index)) in u32:0..M {
    let dist = squared_distance(refrence, vectors[i]);
    if dist < min_dist {
      (dist, i)
    } else {
      (min_dist, min_index)
    }
  }((u32:0xFFFFFFFF, u32:0))
}

// Compute the average squared distance from refrence to all vectors.
fn average_distance(refrence: u8[N], vectors: u8[N][M]) -> u32 {
  let total_dist = for (i, acc) in u32:0..M {
    acc + squared_distance(refrence, vectors[i])
  }(u32:0);
  total_dist >> u32:2  // Divide by M (4 = 2^2)
}

// Compute the distance matrix where element [i][j] is squared_distance(vectors[i], vectors[j]).
fn compute_distance_matrix(vectors: u8[N][M]) -> u32[M][M] {
  let initial_matrix = u32[M][M]:[u32[M]:[0, ...], ...];
  for (i, matrix) in u32:0..M {
    let row = for (j, row_acc): (u32, u32[M]) in u32:0..M {
      update(row_acc, j, squared_distance(vectors[i], vectors[j]))
    }(u32[M]:[0, ...]);
    update(matrix, i, row)
  }(initial_matrix)
}

// Sum all off-diagonal elements of a square matrix.
fn sum_off_diagonals(matrix: u32[M][M]) -> u32 {
  for (i, outer_acc) in u32:0..M {
    let row_sum = for (j, inner_acc) in u32:0..M {
      if i == j {
        inner_acc
      } else {
        inner_acc + matrix[i][j]
      }
    }(u32:0);
    outer_acc + row_sum
  }(u32:0)
}

// Compute variance estimate (average off-diagonal value) from the distance matrix.
fn variance_estimate(matrix: u32[M][M]) -> u32 {
  let total = sum_off_diagonals(matrix);
  total / COUNT_OFF_DIAGONAL
}

// Find the farthest pair from the distance matrix (i < j).
fn farthest_pair_from_matrix(matrix: u32[M][M]) -> (u32, u32, u32) {
  for (i, outer_acc): (u32, (u32, u32, u32)) in u32:0..M {
    let row_result = for (j, inner_acc): (u32, (u32, u32, u32)) in u32:0..M {
      if i < j {
        let d = matrix[i][j];
        let (max_d, idx1, idx2) = inner_acc;
        if d > max_d {
          (d, i, j)
        } else {
          (max_d, idx1, idx2)
        }
      } else {
        inner_acc
      }
    }((u32:0, u32:0, u32:0));
    let (max_d_outer, _idx1_outer, _idx2_outer) = outer_acc;
    let (max_d_row, idx1_row, idx2_row) = row_result;
    if max_d_row > max_d_outer {
      (max_d_row, idx1_row, idx2_row)
    } else {
      outer_acc
    }
  }((u32:0, u32:0, u32:0))
}

// Compute cosine similarity (custom formula).
fn cosine_similarity(a: u8[N], b: u8[N]) -> u32 {
  let dp = dot_product(a, b);
  let mag_a = squared_magnitude(a);
  let mag_b = squared_magnitude(b);
  (dp * dp) / (mag_a * mag_b)
}

// Compute the average cosine similarity over all distinct pairs.
fn average_cosine_similarity(vectors: u8[N][M]) -> u32 {
  let total = for (i, acc) in u32:0..M {
    let row_sum = for (j, inner_acc) in u32:0..M {
      if i < j {
        inner_acc + cosine_similarity(vectors[i], vectors[j])
      } else {
        inner_acc
      }
    }(u32:0);
    acc + row_sum
  }(u32:0);
  total / PAIR_COUNT
}

// Compute the average Manhattan distance from refrence to all vectors.
fn average_manhattan_distance(refrence: u8[N], vectors: u8[N][M]) -> u32 {
  let total = for (i, acc) in u32:0..M {
    acc + manhattan_distance(refrence, vectors[i])
  }(u32:0);
  total / M
}

// High-level analysis: compute the distance matrix and its variance.
fn analyze_vectors(vectors: u8[N][M]) -> (u32[M][M], u32) {
  let matrix = compute_distance_matrix(vectors);
  let variance = variance_estimate(matrix);
  (matrix, variance)
}

// ------------------------------------------------------------
// Mathematical functions: sqrt and standard deviation
// ------------------------------------------------------------

// Approximate square root using 8 iterations of Newton-Raphson.
fn approx_sqrt(x: u32) -> u32 {
  if x == u32:0 {
    u32:0
  } else {
    let initial_guess = x;
    for (_, acc) in u32:0..u32:8 {
      (acc + (x / acc)) / u32:2
    }(initial_guess)
  }
}

// Compute per-component standard deviation.
fn component_stddev(vectors: u8[N][M]) -> u32[N] {
  let avg_vec = average_vector(vectors);
  let initial_stddev = u32[N]:[0, ...];
  for (j, result): (u32, u32[N]) in u32:0..N {
    let sum_sq = for (i, acc) in u32:0..M {
      let diff = if vectors[i][j] > avg_vec[j] {
                   vectors[i][j] - avg_vec[j]
                 } else {
                   avg_vec[j] - vectors[i][j]
                 };
      let sq = (diff as u16) * (diff as u16);
      acc + (sq as u32)
    }(u32:0);
    update(result, j, approx_sqrt(sum_sq / M))
  }(initial_stddev)
}

// Compute global standard deviation as the average of per-component stddev.
fn global_stddev(vectors: u8[N][M]) -> u32 {
  let stddev_arr = component_stddev(vectors);
  let total = for (j, acc) in u32:0..N {
    acc + stddev_arr[j]
  }(u32:0);
  total / N
}

// ------------------------------------------------------------
// Helper functions
// ------------------------------------------------------------

fn compute_nearest_info(refrence: u8[N], vectors: u8[N][M]) -> (u32, u32, u32, u32, u32) {
  let (min_index, min_dist) = find_nearest(refrence, vectors);
  let nearest_vector = vectors[min_index];
  let dot_prod = dot_product(refrence, nearest_vector);
  let avg_dist = average_distance(refrence, vectors);
  let refrence_magnitude = squared_magnitude(refrence);
  (min_index, min_dist, dot_prod, avg_dist, refrence_magnitude)
}

fn compute_arithmetic_results(refrence: u8[N], nearest_vector: u8[N], vectors: u8[N][M])
    -> (u8[N], u8[N], u8[N], u8[N]) {
  let added_vector = vector_add(refrence, nearest_vector);
  let subtracted_vector = vector_subtract(refrence, nearest_vector);
  let scaled_vector = vector_scale(refrence, u8:2);
  let avg_vector = average_vector(vectors);
  (added_vector, subtracted_vector, scaled_vector, avg_vector)
}

fn compute_global_stats(vectors: u8[N][M]) -> (u32[M][M], u32) {
  let (dist_matrix, var_est) = analyze_vectors(vectors);
  (dist_matrix, var_est)
}

fn compute_similarity_stats(refrence: u8[N], nearest_vector: u8[N], vectors: u8[N][M]) -> (u32, u32) {
  let cos_sim = cosine_similarity(refrence, nearest_vector);
  let avg_manhattan = average_manhattan_distance(refrence, vectors);
  (cos_sim, avg_manhattan)
}

fn compute_stddev_info(vectors: u8[N][M]) -> (u32[N], u32) {
  let stddev_components = component_stddev(vectors);
  let global_std = global_stddev(vectors);
  (stddev_components, global_std)
}

fn vector_core(refrence: u8[N], vectors: u8[N][M]) ->
  ((u32, u32, u32, u32, u32),
   (u8[N], u8[N], u8[N], u8[N]),
   (u32[M][M], u32),
   (u32, u32),
   (u32[N], u32)) {
  let nearest_info = compute_nearest_info(refrence, vectors);
  let (min_index, _min_dist, _dot_prod, _avg_dist, _refrence_magnitude) = nearest_info;
  let nearest_vector = vectors[min_index];
  let arithmetic_results = compute_arithmetic_results(refrence, nearest_vector, vectors);
  let global_stats = compute_global_stats(vectors);
  let similarity_stats = compute_similarity_stats(refrence, nearest_vector, vectors);
  let stddev_info = compute_stddev_info(vectors);
  (nearest_info, arithmetic_results, global_stats, similarity_stats, stddev_info)
}
