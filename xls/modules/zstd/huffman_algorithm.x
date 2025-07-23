// Copyright 2025 The XLS Authors
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
//
//
// This file contains huffman algorithm implementation
// The algorithm is based on ZSTD implementation (https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L1273)
//
//    1. Build an initial tree based on the frequencies, sort it in descending order based on frequencies
//    2. Build an unlimited-depth Huffman tree
//    3. Enforce max-depth=11
//    4. Convert the tree to encoding
// }

import std;

const MAX_HUFFMAN_BITS = u32:11;
type HuffmanPrefix = uN[MAX_HUFFMAN_BITS];

pub struct HuffmanPrefixEntry {
    nbits: u4,
    encoding: HuffmanPrefix
}

// TODO: change MAX_SYMBOL_TYPE to u32 when it's supported
// NOTE: MAX_SYMBOL is generic for testing purposes
struct HuffmanPrefixTable<MAX_SYMBOL: u32> {
    data: HuffmanPrefixEntry[MAX_SYMBOL + u32:1]
}

struct FrequencyTable<ADDR_W: u32, MAX_SYMBOL: u32> {
    data:  uN[ADDR_W][MAX_SYMBOL + u32:1]
}

struct HuffmanNode<ADDR_W: u32> {
    symbol: u8,
    freq: uN[ADDR_W],
    parent: uN[ADDR_W]
}

struct HuffmanTree<ADDR_W: u32, MAX_SYMBOL: u32> {
    data: HuffmanNode<ADDR_W>[MAX_SYMBOL + u32:1] // zero
}

fn huffman_first_nonzero_rank<ADDR_W: u32, MAX_SYMBOL: u32>(tree: HuffmanTree<ADDR_W, MAX_SYMBOL>) -> u32 {
    const TABLE_LENGTH = MAX_SYMBOL + u32:1;

    let huff_node = tree.data;
    for (i, max_ix) in u32:0..TABLE_LENGTH {
        let ix = MAX_SYMBOL - i;
        if huff_node[max_ix].freq != u32:0 {
            max_ix
        } else if huff_node[ix].freq != u32:0 {
            ix
        } else {
            max_ix
        }
    }(MAX_SYMBOL)
}

fn huffman_max_nbits<MAX_SYMBOL: u32>(table: HuffmanPrefixTable<MAX_SYMBOL>) -> u4 {
    for (e, max) in table.data {
        if e.nbits > max {
            e.nbits
        } else {
            max
        }
    }(u4:0)
}

fn huffman_tree_to_table<ADDR_W: u32, MAX_SYMBOL: u32>(
    leafs: HuffmanTree<ADDR_W, MAX_SYMBOL>,
    leafs_len: u32,
    nonleafs: HuffmanTree<ADDR_W, MAX_SYMBOL>,
    nonleafs_len: u32
) -> HuffmanPrefixTable<MAX_SYMBOL> {
    type Table = HuffmanPrefixTable<MAX_SYMBOL>;
    type Tree = HuffmanTree<ADDR_W, MAX_SYMBOL>;
    const TABLE_LENGTH = MAX_SYMBOL + u32:1;
    type RankTable = u16[TABLE_LENGTH];

    // https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L709-L714
    let root_table =
    for (i, table) in u32:0..TABLE_LENGTH {
        let ix = MAX_SYMBOL - i;
        if ix >= nonleafs_len {
            table
        } else if ix == nonleafs_len - u32:1 {
            // root = zero bits
            Table{data: update(table.data, ix, HuffmanPrefixEntry{nbits: u4:0, encoding: HuffmanPrefix:0})}
        } else {
            let parent = table.data[nonleafs.data[ix].parent];
            Table{data: update(table.data, ix, HuffmanPrefixEntry{nbits: parent.nbits + u4:1, encoding: HuffmanPrefix:0})}
        }
    }(zero!<Table>());

    let (prefix_table, nb_per_rank) =
    for (i, (table, nb_per_rank)) in u32:0..TABLE_LENGTH {
        if i >= leafs_len {
           (table, nb_per_rank)
        } else {
            let nbits = root_table.data[leafs.data[i].parent].nbits + u4:1;
            let my_symbol = leafs.data[i].symbol;
            (
                Table{data: update(table.data, my_symbol, HuffmanPrefixEntry{nbits: nbits, encoding: HuffmanPrefix:0})},
                update(nb_per_rank, nbits, nb_per_rank[nbits] + u16:1)
            )
        }
    }((zero!<Table>(), zero!<RankTable>()));

    // https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L741-L747
    let (val_per_rank, _) =
    for (i, (val_per_rank, min)) in u32:0..MAX_SYMBOL {
        let n = MAX_SYMBOL - i;
        let next_min = (min + nb_per_rank[n]) >> 1;
        (update(val_per_rank, n, min), next_min)
    }((zero!<RankTable>(), u16:0));

    let (prefix_table, _) =
    for (i, (table, val_per_rank)) in u32:0..TABLE_LENGTH {
        let nbits = table.data[i].nbits;
        let encoding = val_per_rank[nbits] as HuffmanPrefix;

        (
            Table{data: update(table.data, i, HuffmanPrefixEntry{nbits: nbits, encoding: encoding})},
            update(val_per_rank, nbits, val_per_rank[nbits] + u16:1)
        )
    }((prefix_table, val_per_rank));
    prefix_table
}

// https://github.com/facebook/zstd/blob/f9a6031963dee08620855545bdad7d519c208e8a/lib/compress/huf_compress.c#L681
// returns leafs, leaf set size, nonleafs, nonleaf set size
fn huffman_build_tree<ADDR_W: u32, MAX_SYMBOL: u32>(leafs: HuffmanTree<ADDR_W, MAX_SYMBOL>) -> (HuffmanTree<ADDR_W, MAX_SYMBOL>, u32,  HuffmanTree<ADDR_W, MAX_SYMBOL>, u32) {
    const TABLE_LENGTH = MAX_SYMBOL + u32:1;
    const STARTNODE = u32:0; // here it differs, ZSTD places everything in "huffNode" buffer
    type Addr = uN[ADDR_W];
    type Tree = HuffmanTree<ADDR_W, MAX_SYMBOL>;

    // https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L684-L693
    let nonnull_rank = huffman_first_nonzero_rank<ADDR_W, MAX_SYMBOL>(leafs);

    if nonnull_rank == u32:0 {
        (leafs, u32:1, zero!<Tree>(), u32:0)
    } else {
        let node_nb = STARTNODE;
        let nonleafs = zero!<Tree>();
        let low_n = node_nb;
        let low_s = nonnull_rank;
        let node_root = low_s - u32:1;

        // https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L694-L696
        // this tree can be of the same size as leafs since a full binary tree contains n leafs and n-1 non-leafs
        let nonleafs = Tree { data: update(nonleafs.data, node_nb,
            HuffmanNode {
                freq: leafs.data[low_s].freq + leafs.data[low_s-u32:1].freq,
                symbol: u8:0,
                parent: Addr:0
            }
        )};
        let leafs = Tree { data: update(leafs.data, low_s,
            HuffmanNode {
                parent: node_nb,
                ..leafs.data[low_s]
            }
        )};
        let leafs = Tree { data: update(leafs.data, low_s-u32:1,
            HuffmanNode {
                parent: node_nb,
                ..leafs.data[low_s-u32:1]
            }
        )};
        let node_nb = node_nb + u32:1;
        let low_s = low_s - u32:2;

        // https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L697
        let nonleafs =
        for (i, t) in u32:0..TABLE_LENGTH {
            if i < node_nb || i > node_root {
                t
            } else {
                Tree{data: update(t.data, i, HuffmanNode {
                    freq: all_ones!<Addr>(),
                    symbol: u8:0,
                    parent: Addr:0
                })}
            }
        }(nonleafs);

        // https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L700-L707
        let (nonleafs, leafs, node_nb, low_s, low_n) =
        for (_, (nonleafs, leafs, node_nb, low_s, low_n)) in u32:0..TABLE_LENGTH {
            trace_fmt!("{} {} {}", node_nb, low_s, low_n);

            if node_nb <= node_root {
                let (n1, n2, low_s, low_n, first_in_leafs, second_in_leafs) = if low_s != all_ones!<u32>() {
                    let (n1, low_s, low_n, first_in_leafs) = if leafs.data[low_s].freq < nonleafs.data[low_n].freq {
                        (low_s, low_s - u32:1, low_n, true)
                    } else {
                        (low_n, low_s, low_n + u32:1, false)
                    };

                    let (n2, low_s, low_n, second_in_leafs) =
                    if low_s == all_ones!<u32>() {
                        (low_n, low_s, low_n + u32:1, false)
                    } else {
                        if leafs.data[low_s].freq < nonleafs.data[low_n].freq {
                            (low_s, low_s - u32:1, low_n, true)
                        } else {
                            (low_n, low_s, low_n + u32:1, false)
                        }
                    };

                    (n1, n2, low_s, low_n, first_in_leafs, second_in_leafs)
                } else {
                    (low_n, low_n + u32:1, low_s, low_n + u32:2, false, false)
                };

                let freq1 = if first_in_leafs { leafs.data[n1].freq } else { nonleafs.data[n1].freq };
                let freq2 = if second_in_leafs { leafs.data[n2].freq } else { nonleafs.data[n2].freq };
                let first_tree = if first_in_leafs { u32:0 } else { u32:1 };
                let second_tree = if second_in_leafs { u32:0 } else { u32:1 };
                trace_fmt!("Huffman algorithm: joining [{}]:{} <-> [{}]:{}", first_tree, n1, second_tree, n2);


                let nonleafs = Tree { data: update(nonleafs.data, node_nb,
                    HuffmanNode {
                        freq: freq1 + freq2,
                        symbol: u8:0,
                        parent: Addr:0
                    }
                )};

                let (leafs, nonleafs) = if first_in_leafs {
                    (
                        Tree { data: update(leafs.data, n1,
                            HuffmanNode {
                                parent: node_nb,
                                ..leafs.data[n1]
                            }
                        )},
                        nonleafs
                    )
                } else {
                    (
                        leafs,
                        Tree { data: update(nonleafs.data, n1,
                            HuffmanNode {
                                parent: node_nb,
                                ..nonleafs.data[n1]
                            }
                        )}
                    )
                };

                let (leafs, nonleafs) = if second_in_leafs {
                    (
                        Tree { data: update(leafs.data, n2,
                            HuffmanNode {
                                parent: node_nb,
                                ..leafs.data[n2]
                            }
                        )},
                        nonleafs
                    )
                } else {
                    (
                        leafs,
                        Tree { data: update(nonleafs.data, n2,
                            HuffmanNode {
                                parent: node_nb,
                                ..nonleafs.data[n2]
                            }
                        )}
                    )
                };

                (nonleafs, leafs, node_nb+u32:1, low_s, low_n)
            } else {
                (nonleafs, leafs, node_nb, low_s, low_n)
            }
        }((nonleafs, leafs, node_nb, low_s, low_n));
        (leafs, nonnull_rank + u32:1, nonleafs, node_root + u32:1)
    }
}

fn swap_tree_elem<W: u32, N: u32>(t: HuffmanTree<W, N>, i: u32, j: u32) -> HuffmanTree<W, N> {
    HuffmanTree<W, N> {
        data: update(update(t.data, i, t.data[j]), j, t.data[i])
    }
}

fn bitonic_sort_descending_order<SIZE: u32, ADDR_W: u32, N: u32>
    (tree: HuffmanTree<ADDR_W, N>) -> HuffmanTree<ADDR_W, N> {

    assert!(std::is_pow2(SIZE), "For bitonic sorting the array size must be a power of two");
    const LOG_SIZE = std::clog2(SIZE);

    let (sort_tree, _) = for (_, (tree, k)) in u32:0..LOG_SIZE {
        let (sort_tree, _) = for (_, (tree, j)) in u32:0..LOG_SIZE {
            if (j == u32:0) {
                (tree, j)
            } else {
                let sort_tree = for (i, tree) in u32:0..SIZE {
                    let l = i ^ j;
                    if (l <= i) {
                        (tree)
                    } else {
                        if ((((i & k) == u32:0) && (tree.data[i].freq < tree.data[l].freq)) ||
                            (((i & k) != u32:0) && (tree.data[i].freq > tree.data[l].freq))) {
                            swap_tree_elem<ADDR_W,N>(tree, i, l)
                        } else {
                            (tree)
                        }
                    }
                } (tree);
                (sort_tree, j / u32:2)
            }
        } ((tree, k / u32:2));
        (sort_tree, k * u32:2)
    } ((tree, u32:2));
    sort_tree
}


// create a flattened tree with elements in a descending order
fn huffman_initial_tree<ADDR_W: u32, MAX_SYMBOL: u32>(freq: FrequencyTable<ADDR_W, MAX_SYMBOL>) -> HuffmanTree<ADDR_W, MAX_SYMBOL> {
    type Tree = HuffmanTree<ADDR_W, MAX_SYMBOL>;
    type Addr = uN[ADDR_W];
    const TABLE_LENGTH = MAX_SYMBOL + u32:1;

    // first copy everything as is to the tree
    let tree =
    for (i, t) in u32:0..TABLE_LENGTH {
        Tree {
            data: update(t.data, i, HuffmanNode {
                symbol: i as u8,
                freq: freq.data[i],
                parent: Addr:0
            })
        }
    }(zero!<Tree>());

    bitonic_sort_descending_order<TABLE_LENGTH, ADDR_W, MAX_SYMBOL>(tree)
}

fn huffman_bits_to_weight(nbits: u4, max_nbits: u4) -> u4 {
    if nbits == u4:0 {
        u4:0
    } else {
        max_nbits + u4:1 - nbits
    }
}

fn huffman_enforce_max_depth<ADDR_W:u32, MAX_SYMBOL: u32>(
    table: HuffmanPrefixTable<MAX_SYMBOL>,
    max_nbits: u4,
    nbits_bnd: u4
) ->  HuffmanPrefixTable<MAX_SYMBOL> {
    type Table = HuffmanPrefixTable<MAX_SYMBOL>;
    type Tree = HuffmanTree<ADDR_W, MAX_SYMBOL>;
    const TABLE_LENGTH = MAX_SYMBOL + u32:1;
    type RankTable = u16[TABLE_LENGTH];

    let base_cost = u32:1 << (max_nbits - nbits_bnd);

    // step 1: denormalize by enforcing the nbits < nbits_bnd condition
    // https://github.com/facebook/zstd/blob/f9938c217da17ec3e9dcd2a2d99c5cf39536aeb9/lib/compress/huf_compress.c#L393-L397
    let (table_denorm, total_cost) =
    for (i, (table, total_cost)) in u32:0..TABLE_LENGTH {
        if table.data[i].nbits > nbits_bnd {
            let this_cost = base_cost - (u32:1 << (max_nbits - table.data[i].nbits));
            (
                Table{data: update(table.data, i, HuffmanPrefixEntry{nbits: nbits_bnd, ..table.data[i]})},
                total_cost + this_cost
            )
        } else {
            (table, total_cost)
        }
    }((table, u32:0));

    // step 2: renormalize
    // simplified version compared to ZSTD, creates an unoptimal, but correct Huffman dictionary
    // it assumes 2^(nbits_bnd) leafs with nbits=nbits_bnd and recreates the encoding under that assumption
    let nb_per_rank =
    for (i, nb_per_rank) in u32:0..TABLE_LENGTH {
        let nbits = table_denorm.data[i].nbits;
        if nbits == u4:0 {
            nb_per_rank
        } else {
            update(nb_per_rank, nbits, nb_per_rank[nbits] + u16:1)
        }
    }(zero!<RankTable>());

    let (val_per_rank, _) =
    for (i, (val_per_rank, min)) in u32:0..MAX_SYMBOL {
        let n = MAX_SYMBOL - i;
        trace_fmt!("min {}: {}", i, min);
        let rank_cnt = if n == nbits_bnd as u32 {
            u16:1 << (n - u32:1)
        } else {
            nb_per_rank[n]
        };

        let next_min = (min + rank_cnt) >> 1;
        (update(val_per_rank, n, min), next_min)
    }((zero!<RankTable>(), u16:0));

    let (table_norm, _) =
    for (i, (table, val_per_rank)) in u32:0..TABLE_LENGTH {
        let nbits = table.data[i].nbits;
        let encoding = val_per_rank[nbits] as HuffmanPrefix;

        (
            Table{data: update(table.data, i, HuffmanPrefixEntry{nbits: nbits, encoding: encoding})},
            update(val_per_rank, nbits, val_per_rank[nbits] + u16:1)
        )
    }((table_denorm, val_per_rank));

    table_norm
}

fn huffman_get_prefixes<ADDR_W: u32, MAX_SYMBOL: u32>(freq: FrequencyTable<ADDR_W, MAX_SYMBOL>) -> HuffmanPrefixTable<MAX_SYMBOL> {
    type PrefixTable = HuffmanPrefixTable<MAX_SYMBOL>;
    type FrequencyTable = FrequencyTable<ADDR_W, MAX_SYMBOL>;
    const HUFFMAN_MAX_LOG = u4:11;

    let tree = huffman_initial_tree<ADDR_W, MAX_SYMBOL>(freq);
    let (leafs, leafs_len, nonleafs, nonleafs_len) = huffman_build_tree<ADDR_W, MAX_SYMBOL>(tree);
    let prefix_table = huffman_tree_to_table<ADDR_W, MAX_SYMBOL>(leafs, leafs_len, nonleafs, nonleafs_len);
    let max_nbits = huffman_max_nbits<MAX_SYMBOL>(prefix_table);


    if  max_nbits > HUFFMAN_MAX_LOG {
        // see: https://datatracker.ietf.org/doc/html/rfc8878#section-4.2.1
        huffman_enforce_max_depth<ADDR_W, MAX_SYMBOL>(prefix_table, max_nbits, HUFFMAN_MAX_LOG)
    } else {
        prefix_table
    }
}

const TEST_ADDR_W = u32:32;
const TEST_MAX_SYMBOL = u32:5;

type TestHuffmanTree = HuffmanTree<TEST_ADDR_W, TEST_MAX_SYMBOL>;
type TestHuffmanNode = HuffmanNode<TEST_ADDR_W>;
type TestHuffmanPrefixTable = HuffmanPrefixTable<TEST_MAX_SYMBOL>;
type TestFrequencyTable = FrequencyTable<TEST_ADDR_W, TEST_MAX_SYMBOL>;

struct BuildTreeTestEntry {
    initial_tree: TestHuffmanTree,
    expected_leafs: TestHuffmanTree,
    expected_leafs_len: u32,
    expected_nonleafs: TestHuffmanTree,
    expected_nonleafs_len: u32
}

#[test]
fn huffman_build_tree_test() {
    type Addr = uN[TEST_ADDR_W];
    let test_cases = [
        BuildTreeTestEntry { // testcase 1.
            initial_tree: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:2, freq: Addr:5, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:4, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:3, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:2, parent: Addr:0},
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:0, parent: Addr:0},
            ]},
            expected_leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:2, freq: Addr:5, parent: Addr:2},
                TestHuffmanNode{symbol: u8:1, freq: Addr:4, parent: Addr:2},
                TestHuffmanNode{symbol: u8:4, freq: Addr:3, parent: Addr:1},
                TestHuffmanNode{symbol: u8:3, freq: Addr:2, parent: Addr:0},
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:0, parent: Addr:0},
            ]},
            expected_leafs_len: u32:5,
            expected_nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:3, parent: Addr:1},
                TestHuffmanNode{symbol: u8:0, freq: Addr:6, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:9, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:15, parent: Addr:0},
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>()
            ]},
            expected_nonleafs_len: u32:4
        },
        BuildTreeTestEntry { // testcase 2.
            initial_tree: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:100, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:9, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:6, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:5, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:4, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:100, parent: Addr:4},
                TestHuffmanNode{symbol: u8:3, freq: Addr:9, parent: Addr:2},
                TestHuffmanNode{symbol: u8:5, freq: Addr:6, parent: Addr:2},
                TestHuffmanNode{symbol: u8:2, freq: Addr:5, parent: Addr:1},
                TestHuffmanNode{symbol: u8:1, freq: Addr:4, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs_len: u32:6,
            expected_nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:5, parent: Addr:1},
                TestHuffmanNode{symbol: u8:0, freq: Addr:10, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:15, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:25, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:125, parent: Addr:0},
                zero!<TestHuffmanNode>(),
            ]},
            expected_nonleafs_len: u32:5
        },
        BuildTreeTestEntry { // testcase 3. uniform distribution
            initial_tree: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:2},
                TestHuffmanNode{symbol: u8:1, freq: Addr:1, parent: Addr:2},
                TestHuffmanNode{symbol: u8:2, freq: Addr:1, parent: Addr:1},
                TestHuffmanNode{symbol: u8:3, freq: Addr:1, parent: Addr:1},
                TestHuffmanNode{symbol: u8:4, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs_len: u32:6,
            expected_nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:2, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:2, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:2, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:4, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:6, parent: Addr:0},
                zero!<TestHuffmanNode>(),
            ]},
            expected_nonleafs_len: u32:5
        },
        BuildTreeTestEntry { // testcase 4. 2^k
            initial_tree: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:32, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:16, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:8, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:4, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:2, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:32, parent: Addr:4},
                TestHuffmanNode{symbol: u8:1, freq: Addr:16, parent: Addr:3},
                TestHuffmanNode{symbol: u8:2, freq: Addr:8, parent: Addr:2},
                TestHuffmanNode{symbol: u8:3, freq: Addr:4, parent: Addr:1},
                TestHuffmanNode{symbol: u8:4, freq: Addr:2, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs_len: u32:6,
            expected_nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:3, parent: Addr:1},
                TestHuffmanNode{symbol: u8:0, freq: Addr:7, parent: Addr:2},
                TestHuffmanNode{symbol: u8:0, freq: Addr:15, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:31, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:63, parent: Addr:0},
                zero!<TestHuffmanNode>(),
            ]},
            expected_nonleafs_len: u32:5
        },
        BuildTreeTestEntry { // testcase 5. single symbol
            initial_tree: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:0, parent: Addr:2},
                TestHuffmanNode{symbol: u8:1, freq: Addr:0, parent: Addr:2},
                TestHuffmanNode{symbol: u8:2, freq: Addr:0, parent: Addr:1},
                TestHuffmanNode{symbol: u8:3, freq: Addr:0, parent: Addr:1},
                TestHuffmanNode{symbol: u8:4, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:1, parent: Addr:0},
            ]},
            expected_leafs_len: u32:6,
            expected_nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:0, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:0, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:0},
                zero!<TestHuffmanNode>(),
            ]},
            expected_nonleafs_len: u32:5
        },
        BuildTreeTestEntry { // testcase 6. minimal tree
            initial_tree: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:3, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:2, parent: Addr:0},
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
            ]},
            expected_leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:3, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:2, parent: Addr:0},
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>()
            ]},
            expected_leafs_len: u32:2,
            expected_nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:5, parent: Addr:0},
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>(),
            ]},
            expected_nonleafs_len: u32:1
        },
        BuildTreeTestEntry { // testcase 7. unbalanced tree
            initial_tree: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:5, freq: Addr:45, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:16, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:13, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:12, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:9, parent: Addr:0},
                TestHuffmanNode{symbol: u8:0, freq: Addr:5, parent: Addr:0},
            ]},
            expected_leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:5, freq: Addr:45, parent: Addr:4},
                TestHuffmanNode{symbol: u8:4, freq: Addr:16, parent: Addr:2},
                TestHuffmanNode{symbol: u8:3, freq: Addr:13, parent: Addr:1},
                TestHuffmanNode{symbol: u8:2, freq: Addr:12, parent: Addr:1},
                TestHuffmanNode{symbol: u8:1, freq: Addr:9, parent: Addr:0},
                TestHuffmanNode{symbol: u8:0, freq: Addr:5, parent: Addr:0},
            ]},
            expected_leafs_len: u32:6,
            expected_nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:14, parent: Addr:2},
                TestHuffmanNode{symbol: u8:0, freq: Addr:25, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:30, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:55, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:100, parent: Addr:0},
                zero!<TestHuffmanNode>(),
            ]},
            expected_nonleafs_len: u32:5
        }
    ];

    for ((i, entry), _) in enumerate(test_cases) {
        let (leafs, leafs_len, nonleafs, nonleafs_len) = huffman_build_tree<TEST_ADDR_W, TEST_MAX_SYMBOL>(entry.initial_tree);
        assert_eq(leafs, entry.expected_leafs);
        assert_eq(leafs_len, entry.expected_leafs_len);
        assert_eq(nonleafs, entry.expected_nonleafs);
        assert_eq(nonleafs_len, entry.expected_nonleafs_len);
        trace_fmt!("[TEST] huffman_build_tree_test::{} passed", i);
    }(());
}

struct TreeToTableTestEntry {
    leafs: TestHuffmanTree,
    leafs_len: u32,
    nonleafs: TestHuffmanTree,
    nonleafs_len: u32,
    expected_table: TestHuffmanPrefixTable,
    expected_max_nbits: u4
}

#[test]
fn huffman_tree_to_table_test(){
    type Addr = uN[TEST_ADDR_W];

    let test_cases = [
        TreeToTableTestEntry {
            leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:2, freq: Addr:5, parent: Addr:2},
                TestHuffmanNode{symbol: u8:1, freq: Addr:4, parent: Addr:2},
                TestHuffmanNode{symbol: u8:4, freq: Addr:3, parent: Addr:1},
                TestHuffmanNode{symbol: u8:3, freq: Addr:2, parent: Addr:0},
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:0, parent: Addr:0},
            ]},
            leafs_len: u32:5,
            nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:3, parent: Addr:1},
                TestHuffmanNode{symbol: u8:0, freq: Addr:6, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:9, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:15, parent: Addr:0},
                zero!<TestHuffmanNode>(),
                zero!<TestHuffmanNode>()
            ]},
            nonleafs_len: u32:4,
            expected_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b000 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b01 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b10 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b001 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b11 },
                    zero!<HuffmanPrefixEntry>()
                ]
            },
            expected_max_nbits: u4:3
        },
        TreeToTableTestEntry {
            leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:100, parent: Addr:4},
                TestHuffmanNode{symbol: u8:3, freq: Addr:9, parent: Addr:2},
                TestHuffmanNode{symbol: u8:5, freq: Addr:6, parent: Addr:2},
                TestHuffmanNode{symbol: u8:2, freq: Addr:5, parent: Addr:1},
                TestHuffmanNode{symbol: u8:1, freq: Addr:4, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:1, parent: Addr:0},
            ]},
            leafs_len: u32:6,
            nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:5, parent: Addr:1},
                TestHuffmanNode{symbol: u8:0, freq: Addr:10, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:15, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:25, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:125, parent: Addr:0},
                zero!<TestHuffmanNode>(),
            ]},
            nonleafs_len: u32:5,
            expected_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:1, encoding: HuffmanPrefix:0b1 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0000 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b001 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b010 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0001 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b011 },
                ]
            },
            expected_max_nbits: u4:4
        },
        TreeToTableTestEntry {
            leafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:2},
                TestHuffmanNode{symbol: u8:1, freq: Addr:1, parent: Addr:2},
                TestHuffmanNode{symbol: u8:2, freq: Addr:1, parent: Addr:1},
                TestHuffmanNode{symbol: u8:3, freq: Addr:1, parent: Addr:1},
                TestHuffmanNode{symbol: u8:4, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:1, parent: Addr:0},
            ]},
            leafs_len: u32:6,
            nonleafs: TestHuffmanTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:2, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:2, parent: Addr:3},
                TestHuffmanNode{symbol: u8:0, freq: Addr:2, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:4, parent: Addr:4},
                TestHuffmanNode{symbol: u8:0, freq: Addr:6, parent: Addr:0},
                zero!<TestHuffmanNode>(),
            ]},
            nonleafs_len: u32:5,
            expected_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b10 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b11 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b000 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b001 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b010 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b011 },
                ]
            },
            expected_max_nbits: u4:3
        },
    ];
    for ((i, entry), _) in enumerate(test_cases) {
        let prefix_table = huffman_tree_to_table<TEST_ADDR_W, TEST_MAX_SYMBOL>(entry.leafs, entry.leafs_len, entry.nonleafs, entry.nonleafs_len);
        let max_nbits = huffman_max_nbits<TEST_MAX_SYMBOL>(prefix_table);
        assert_eq(entry.expected_table, prefix_table);
        assert_eq(entry.expected_max_nbits, max_nbits);
        trace_fmt!("[TEST] huffman_tree_to_table_test::{} passed", i);
    }(());
}

struct MaxDepthTestEntry {
    input_table: TestHuffmanPrefixTable,
    nbits_bnd: u4,
    expected_table: TestHuffmanPrefixTable
}

#[test]
fn huffman_enforce_max_depth_test() {
    type Addr = uN[TEST_ADDR_W];

    let test_cases = [
       MaxDepthTestEntry {
            input_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:12, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:11, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:10, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:11, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:13, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:1, encoding: HuffmanPrefix:0 },
                ]
            },
            nbits_bnd: u4:5,
            expected_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:5, encoding: HuffmanPrefix:0b00000 },
                    HuffmanPrefixEntry { nbits: u4:5, encoding: HuffmanPrefix:0b00001 },
                    HuffmanPrefixEntry { nbits: u4:5, encoding: HuffmanPrefix:0b00010 },
                    HuffmanPrefixEntry { nbits: u4:5, encoding: HuffmanPrefix:0b00011 },
                    HuffmanPrefixEntry { nbits: u4:5, encoding: HuffmanPrefix:0b00100 },
                    HuffmanPrefixEntry { nbits: u4:1, encoding: HuffmanPrefix:0b1 },
                ]
            }
        },
        MaxDepthTestEntry {
            input_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:12, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:11, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:10, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:11, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:13, encoding: HuffmanPrefix:0 },
                    HuffmanPrefixEntry { nbits: u4:1, encoding: HuffmanPrefix:0 },
                ]
            },
            nbits_bnd: u4:4,
            expected_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0000 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0001 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0010 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0011 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0100 },
                    HuffmanPrefixEntry { nbits: u4:1, encoding: HuffmanPrefix:0b1 },
                ]
            }
        },
        MaxDepthTestEntry {
            input_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b11 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0000 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b001 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b010 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b10 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b011 },
                ]
            },
            nbits_bnd: u4:3,
            expected_table: TestHuffmanPrefixTable{
                data: [
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b10 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b000 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b001 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b010 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b11 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b011 },
                ]
            }
        }
    ];

    for ((i, entry), _) in enumerate(test_cases) {
        let max_nbits = huffman_max_nbits<TEST_MAX_SYMBOL>(entry.input_table);
        assert_eq(entry.expected_table, huffman_enforce_max_depth<TEST_ADDR_W, TEST_MAX_SYMBOL>(entry.input_table, max_nbits, entry.nbits_bnd));
        trace_fmt!("[TEST] huffman_enforce_max_depth_test::{} passed", i);
    }(());
}

// The biggest symbol has been increased to a power of two minus one
// due to the limitations of bitonic sorting.
const TEST_MAX_SYMBOL_INIT_TREE = u32:7;

type TestHuffmanTreeInitTree = HuffmanTree<TEST_ADDR_W, TEST_MAX_SYMBOL_INIT_TREE>;
type TestFrequencyTableInitTree = FrequencyTable<TEST_ADDR_W, TEST_MAX_SYMBOL_INIT_TREE>;
type TestHuffmanPrefixTableInitTree = HuffmanPrefixTable<TEST_MAX_SYMBOL_INIT_TREE>;

#[test]
fn huffman_initial_tree_test() {
    type Addr = uN[TEST_ADDR_W];
    let test_cases = [
        (
            TestHuffmanTreeInitTree{data: [
                TestHuffmanNode{symbol: u8:7, freq: Addr:7, parent: Addr:0},
                TestHuffmanNode{symbol: u8:6, freq: Addr:6, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:5, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:4, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:3, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:2, parent: Addr:0},
                TestHuffmanNode{symbol: u8:0, freq: Addr:1, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:0, parent: Addr:0},
            ]},
            TestFrequencyTableInitTree{data: [Addr:1, Addr:4, Addr:5, Addr:2, Addr:3, Addr:0, Addr:6, Addr:7]},
        ),
        (
            TestHuffmanTreeInitTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:100, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:66, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:22, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:9, parent: Addr:0},
                TestHuffmanNode{symbol: u8:7, freq: Addr:6, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:5, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:4, parent: Addr:0},
                TestHuffmanNode{symbol: u8:6, freq: Addr:1, parent: Addr:0},
            ]},
            TestFrequencyTableInitTree{data: [Addr:100, Addr:66, Addr:22, Addr:4, Addr:5, Addr:9, Addr:1, Addr:6]},
        ),
        (
            TestHuffmanTreeInitTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:6, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:7, freq: Addr:0, parent: Addr:0},
            ]},
            TestFrequencyTableInitTree{data: [Addr:0, Addr:0, Addr:0, Addr:0, Addr:0, Addr:0, Addr:0, Addr:0]},
        ),
        (
            TestHuffmanTreeInitTree{data: [
                TestHuffmanNode{symbol: u8:0, freq: Addr:15, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:10, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:5, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:6, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:7, freq: Addr:0, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:0, parent: Addr:0},
            ]},
            TestFrequencyTableInitTree{data: [Addr:15, Addr:10, Addr:0, Addr:0, Addr:0, Addr:5, Addr:0, Addr:0]},
        ),
        (
            TestHuffmanTreeInitTree{data: [
                TestHuffmanNode{symbol: u8:7, freq: Addr:45, parent: Addr:0},
                TestHuffmanNode{symbol: u8:6, freq: Addr:40, parent: Addr:0},
                TestHuffmanNode{symbol: u8:5, freq: Addr:35, parent: Addr:0},
                TestHuffmanNode{symbol: u8:4, freq: Addr:16, parent: Addr:0},
                TestHuffmanNode{symbol: u8:3, freq: Addr:13, parent: Addr:0},
                TestHuffmanNode{symbol: u8:2, freq: Addr:12, parent: Addr:0},
                TestHuffmanNode{symbol: u8:1, freq: Addr:9, parent: Addr:0},
                TestHuffmanNode{symbol: u8:0, freq: Addr:5, parent: Addr:0},
            ]},
            TestFrequencyTableInitTree{data: [Addr:5, Addr:9, Addr:12, Addr:13, Addr:16, Addr:35, Addr:40, Addr:45]}
        )
    ];

    for ((i, (expected_tree, frequencies)), _) in enumerate(test_cases) {
        assert_eq(expected_tree, huffman_initial_tree<TEST_ADDR_W, TEST_MAX_SYMBOL_INIT_TREE>(frequencies));
        trace_fmt!("[TEST] huffman_initial_tree_test::{} passed", i);
    }(());
}

#[test]
fn huffman_bits_to_weight_test() {
    // testcase directly from RFC https://datatracker.ietf.org/doc/html/rfc8878#section-4.2.1
    type Addr = uN[TEST_ADDR_W];
    const MAX_NBITS = u4:4;

    let test_cases = [
        (u4:1, u4:4),
        (u4:2, u4:3),
        (u4:3, u4:2),
        (u4:0, u4:0),
        (u4:4, u4:1),
    ];

    for ((i, (nbits, expected_weight)), _) in enumerate(test_cases) {
        assert_eq(expected_weight, huffman_bits_to_weight(nbits, MAX_NBITS));
        trace_fmt!("[TEST] huffman_bits_to_weight_test::{} passed", i);
    }(());
}


#[test]
fn huffman_get_prefixes_test() {
    type Addr = uN[TEST_ADDR_W];
    let test_cases = [
        (
            TestHuffmanPrefixTableInitTree{
                data: [
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0000 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b001 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b010 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b10 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b11 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0001 },
                    HuffmanPrefixEntry { nbits: u4:3, encoding: HuffmanPrefix:0b011 },
                    zero!<HuffmanPrefixEntry>()
                ]
            },
            TestFrequencyTableInitTree{data: [Addr:1, Addr:4, Addr:5, Addr:6, Addr:7, Addr:2, Addr:3, Addr:0]}
        ),
        (
            TestHuffmanPrefixTableInitTree{
                data: [
                    HuffmanPrefixEntry { nbits: u4:5, encoding: HuffmanPrefix:0b00000 },
                    HuffmanPrefixEntry { nbits: u4:5, encoding: HuffmanPrefix:0b00001 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0001 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0010 },
                    HuffmanPrefixEntry { nbits: u4:4, encoding: HuffmanPrefix:0b0011 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b01 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b10 },
                    HuffmanPrefixEntry { nbits: u4:2, encoding: HuffmanPrefix:0b11 },
                ]
            },
            TestFrequencyTableInitTree{data: [Addr:5, Addr:9, Addr:12, Addr:13, Addr:16, Addr:35, Addr:40, Addr:45]}
        )
    ];

    for ((i, (expected_prefixes, frequencies)), _) in enumerate(test_cases) {
        assert_eq(expected_prefixes, huffman_get_prefixes<TEST_ADDR_W, TEST_MAX_SYMBOL_INIT_TREE>(frequencies));
        trace_fmt!("[TEST] huffman_get_prefixes_test::{} passed", i);
    }(());
}

// Instances for codegen
const INST_ADDR_W = u32:32;

// The biggest symbol has to be a power of two minus one
// due to the limitations of bitonic sorting.

// TODO: The target value of INST_MAX_SYMBOL is 255, but it has been temporarily
// set to a lower value because it is still a work in progress.
const INST_MAX_SYMBOL = u32:15;

fn huffman_initial_tree_inst(freq: FrequencyTable<INST_ADDR_W, INST_MAX_SYMBOL>) -> HuffmanTree<INST_ADDR_W, INST_MAX_SYMBOL> {
    huffman_initial_tree<INST_ADDR_W, INST_MAX_SYMBOL>(freq)
}

fn huffman_get_prefixes_inst(freq: FrequencyTable<INST_ADDR_W, INST_MAX_SYMBOL>) -> HuffmanPrefixTable<INST_MAX_SYMBOL> {
    huffman_get_prefixes<INST_ADDR_W, INST_MAX_SYMBOL>(freq)
}

fn huffman_build_tree_inst(leafs: HuffmanTree<INST_ADDR_W, INST_MAX_SYMBOL>) -> (HuffmanTree<INST_ADDR_W, INST_MAX_SYMBOL>, u32,  HuffmanTree<INST_ADDR_W, INST_MAX_SYMBOL>, u32) {
   huffman_build_tree<INST_ADDR_W, INST_MAX_SYMBOL>(leafs)
}

fn huffman_tree_to_table_inst(leafs: HuffmanTree<INST_ADDR_W, INST_MAX_SYMBOL>, leafs_len: u32, nonleafs: HuffmanTree<INST_ADDR_W, INST_MAX_SYMBOL>, nonleafs_len: u32) -> HuffmanPrefixTable<INST_MAX_SYMBOL> {
    huffman_tree_to_table<INST_ADDR_W, INST_MAX_SYMBOL>(leafs, leafs_len, nonleafs, nonleafs_len)
}

fn huffman_max_nbits_inst(tab: HuffmanPrefixTable<INST_MAX_SYMBOL>) -> u4 {
    huffman_max_nbits<INST_MAX_SYMBOL>(tab)
}
