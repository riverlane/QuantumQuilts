use std::time::Instant;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::{Rng, SeedableRng, SliceRandom};
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

type BitRow = Vec<u64>;
type LocationGroups = Vec<Vec<usize>>;

#[derive(Clone)]
struct LocationOccupancy {
    word_masks: [(usize, u64); 4],
    len: usize,
}

impl LocationOccupancy {
    fn from_columns(columns: impl IntoIterator<Item = usize>) -> Self {
        let mut word_masks = [(0, 0); 4];
        let mut len = 0;
        for column in columns {
            let word = column / 64;
            let mask = 1_u64 << (column % 64);
            if let Some((_, existing_mask)) = word_masks[..len]
                .iter_mut()
                .find(|(existing_word, _)| *existing_word == word)
            {
                *existing_mask |= mask;
            } else {
                word_masks[len] = (word, mask);
                len += 1;
            }
        }
        Self { word_masks, len }
    }

    fn symplectic(group: &[usize], qubits: usize) -> Self {
        Self::from_columns(group.iter().flat_map(|&qubit| [qubit, qubit + qubits]))
    }

    fn binary(group: &[usize]) -> Self {
        Self::from_columns(group.iter().copied())
    }

    fn is_occupied(&self, row: &BitRow) -> bool {
        self.word_masks[..self.len]
            .iter()
            .any(|&(word, mask)| row[word] & mask != 0)
    }
}

type LocationOccupancies = Vec<LocationOccupancy>;

#[pyclass]
#[derive(Clone, Debug)]
pub struct SearchResult {
    #[pyo3(get)]
    pub distance: Option<usize>,
    #[pyo3(get)]
    pub witness: Vec<u8>,
    #[pyo3(get)]
    pub trials_requested: usize,
    #[pyo3(get)]
    pub trials_completed: usize,
    #[pyo3(get)]
    pub hits: usize,
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    #[pyo3(get)]
    pub target_hit: bool,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(distance={:?}, trials_completed={}, hits={}, elapsed_seconds={:.6}, target_hit={})",
            self.distance, self.trials_completed, self.hits, self.elapsed_seconds, self.target_hit
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchMethod {
    Hamming,
    Grouped,
    BinaryGrouped,
    LocalClifford,
    LocationLinear,
    ThreeBlock,
    ThreeBlockGlobal,
}

impl SearchMethod {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "grouped" => Ok(Self::Grouped),
            "local-clifford" => Ok(Self::LocalClifford),
            "location-linear" => Ok(Self::LocationLinear),
            "three-block" => Ok(Self::ThreeBlock),
            "three-block-global" => Ok(Self::ThreeBlockGlobal),
            _ => Err(PyValueError::new_err(format!(
                "unknown symplectic search method {value:?}; expected grouped, local-clifford, location-linear, three-block, or three-block-global"
            ))),
        }
    }
}

#[derive(Clone)]
struct TrialBest {
    distance: usize,
    witness: BitRow,
    hits: usize,
}

impl TrialBest {
    fn empty(words: usize) -> Self {
        Self {
            distance: usize::MAX,
            witness: vec![0; words],
            hits: 0,
        }
    }

    fn consider(&mut self, candidate: &BitRow, weight: usize) {
        if weight == 0 {
            return;
        }
        if weight < self.distance {
            self.distance = weight;
            self.witness.clone_from(candidate);
            self.hits = 1;
        } else if weight == self.distance {
            self.hits += 1;
            if candidate < &self.witness {
                self.witness.clone_from(candidate);
            }
        }
    }

    fn merge(&mut self, other: TrialBest) {
        if other.distance < self.distance {
            *self = other;
        } else if other.distance == self.distance && other.distance != usize::MAX {
            self.hits += other.hits;
            if other.witness < self.witness {
                self.witness = other.witness;
            }
        }
    }
}

fn words_for(ncols: usize) -> usize {
    ncols.div_ceil(64)
}

fn get_bit(row: &BitRow, column: usize) -> bool {
    ((row[column / 64] >> (column % 64)) & 1) != 0
}

fn set_bit(row: &mut BitRow, column: usize) {
    row[column / 64] |= 1_u64 << (column % 64);
}

fn xor_assign(left: &mut BitRow, right: &BitRow) {
    for (a, b) in left.iter_mut().zip(right) {
        *a ^= *b;
    }
}

fn xor_rows(left: &BitRow, right: &BitRow) -> BitRow {
    left.iter().zip(right).map(|(a, b)| a ^ b).collect()
}

fn dot(left: &BitRow, right: &BitRow) -> bool {
    left.iter()
        .zip(right)
        .fold(0_u32, |parity, (a, b)| parity ^ ((a & b).count_ones() & 1))
        != 0
}

fn hamming_weight(row: &BitRow) -> usize {
    row.iter().map(|word| word.count_ones() as usize).sum()
}

fn symplectic_weight(row: &BitRow, qubits: usize) -> usize {
    (0..qubits)
        .filter(|&q| get_bit(row, q) || get_bit(row, q + qubits))
        .count()
}

fn location_weight(row: &BitRow, qubits: usize, groups: &[Vec<usize>]) -> usize {
    groups
        .iter()
        .filter(|group| {
            group
                .iter()
                .any(|&q| get_bit(row, q) || get_bit(row, q + qubits))
        })
        .count()
}

fn build_location_occupancies(
    groups: &LocationGroups,
    qubits: usize,
    method: SearchMethod,
) -> LocationOccupancies {
    groups
        .iter()
        .map(|group| {
            if method == SearchMethod::BinaryGrouped {
                LocationOccupancy::binary(group)
            } else {
                LocationOccupancy::symplectic(group, qubits)
            }
        })
        .collect()
}

fn location_weight_with_occupancies(row: &BitRow, occupancies: &LocationOccupancies) -> usize {
    occupancies
        .iter()
        .filter(|occupancy| occupancy.is_occupied(row))
        .count()
}

fn bounded_location_weight(
    row: &BitRow,
    occupancies: &LocationOccupancies,
    cutoff: usize,
) -> Option<usize> {
    let mut weight = 0;
    for occupancy in occupancies {
        if occupancy.is_occupied(row) {
            weight += 1;
            if weight > cutoff {
                return None;
            }
        }
    }
    Some(weight)
}

fn output_weight(
    row: &BitRow,
    method: SearchMethod,
    qubits: usize,
    location_groups: Option<&LocationGroups>,
) -> usize {
    match method {
        SearchMethod::Hamming => hamming_weight(row),
        _ => location_groups.map_or_else(
            || symplectic_weight(row, qubits),
            |groups| location_weight(row, qubits, groups),
        ),
    }
}

fn working_location_weight(
    row: &BitRow,
    method: SearchMethod,
    qubits: usize,
    groups: &[Vec<usize>],
) -> usize {
    let blocks = if matches!(
        method,
        SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal
    ) {
        3
    } else {
        2
    };
    groups
        .iter()
        .filter(|group| {
            group
                .iter()
                .any(|&q| (0..blocks).any(|block| get_bit(row, q + block * qubits)))
        })
        .count()
}

fn working_weight(
    row: &BitRow,
    method: SearchMethod,
    qubits: usize,
    location_groups: Option<&LocationGroups>,
    location_occupancies: Option<&LocationOccupancies>,
) -> usize {
    if let Some(groups) = location_groups {
        if !matches!(
            method,
            SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal
        ) {
            return location_weight_with_occupancies(
                row,
                location_occupancies.expect("location occupancies accompany location groups"),
            );
        }
        return working_location_weight(row, method, qubits, groups);
    }
    match method {
        SearchMethod::Hamming => hamming_weight(row),
        SearchMethod::Grouped | SearchMethod::LocalClifford | SearchMethod::LocationLinear => {
            symplectic_weight(row, qubits)
        }
        SearchMethod::BinaryGrouped => hamming_weight(row),
        SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal => hamming_weight(row) / 2,
    }
}

fn normalize_location_groups(
    supplied: Option<Vec<Vec<usize>>>,
    qubits: usize,
) -> PyResult<Option<LocationGroups>> {
    let Some(mut groups) = supplied else {
        return Ok(None);
    };
    let mut seen = vec![false; qubits];
    for group in &mut groups {
        if group.is_empty() || group.len() > 2 {
            return Err(PyValueError::new_err(
                "each location group must contain one or two qubit indices",
            ));
        }
        let supplied_len = group.len();
        group.sort_unstable();
        group.dedup();
        if group.len() != supplied_len {
            return Err(PyValueError::new_err(
                "each location group must contain one or two distinct qubit indices",
            ));
        }
        for &qubit in group.iter() {
            if qubit >= qubits {
                return Err(PyValueError::new_err(format!(
                    "location-group qubit index {qubit} is outside 0..{qubits}",
                )));
            }
            if std::mem::replace(&mut seen[qubit], true) {
                return Err(PyValueError::new_err(format!(
                    "qubit {qubit} occurs in more than one location group",
                )));
            }
        }
    }
    groups.extend(
        seen.iter()
            .enumerate()
            .filter_map(|(qubit, &present)| (!present).then_some(vec![qubit])),
    );
    groups.sort_unstable();
    Ok(Some(groups))
}

fn normalize_binary_location_groups(
    supplied: Option<Vec<Vec<usize>>>,
    ncols: usize,
) -> PyResult<LocationGroups> {
    let mut groups = supplied.unwrap_or_default();
    let mut seen = vec![false; ncols];
    for group in &mut groups {
        if group.is_empty() || group.len() > 4 {
            return Err(PyValueError::new_err(
                "each binary location group must contain one to four column indices",
            ));
        }
        let supplied_len = group.len();
        group.sort_unstable();
        group.dedup();
        if group.len() != supplied_len {
            return Err(PyValueError::new_err(
                "each binary location group must contain distinct column indices",
            ));
        }
        for &column in group.iter() {
            if column >= ncols {
                return Err(PyValueError::new_err(format!(
                    "binary location-group column index {column} is outside 0..{ncols}",
                )));
            }
            if std::mem::replace(&mut seen[column], true) {
                return Err(PyValueError::new_err(format!(
                    "column {column} occurs in more than one binary location group",
                )));
            }
        }
    }
    groups.extend(
        seen.iter()
            .enumerate()
            .filter_map(|(column, &present)| (!present).then_some(vec![column])),
    );
    groups.sort_unstable();
    Ok(groups)
}

fn packed_rows(array: PyReadonlyArray2<'_, u8>) -> PyResult<(Vec<BitRow>, usize)> {
    let view = array.as_array();
    let ncols = view.shape()[1];
    let mut rows = Vec::with_capacity(view.shape()[0]);
    for source in view.rows() {
        let mut row = vec![0_u64; words_for(ncols)];
        for (column, bit) in source.iter().enumerate() {
            if bit & 1 != 0 {
                set_bit(&mut row, column);
            }
        }
        rows.push(row);
    }
    Ok((rows, ncols))
}

fn rref_with_order(rows: &mut Vec<BitRow>, columns: &[usize]) -> Vec<usize> {
    let mut pivots = Vec::with_capacity(rows.len());
    let mut rank = 0;
    for &column in columns {
        let Some(pivot_index) = (rank..rows.len()).find(|&index| get_bit(&rows[index], column))
        else {
            continue;
        };
        rows.swap(rank, pivot_index);
        let pivot = rows[rank].clone();
        for (index, row) in rows.iter_mut().enumerate() {
            if index != rank && get_bit(row, column) {
                xor_assign(row, &pivot);
            }
        }
        pivots.push(column);
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rows.truncate(rank);
    pivots
}

fn natural_rref(rows: &mut Vec<BitRow>, ncols: usize) -> Vec<usize> {
    rref_with_order(rows, &(0..ncols).collect::<Vec<_>>())
}

fn nullspace(rows: &[BitRow], ncols: usize) -> Vec<BitRow> {
    let mut reduced = rows.to_vec();
    let pivots = natural_rref(&mut reduced, ncols);
    let mut is_pivot = vec![false; ncols];
    for &pivot in &pivots {
        is_pivot[pivot] = true;
    }
    let mut basis = Vec::with_capacity(ncols - pivots.len());
    for free in (0..ncols).filter(|&column| !is_pivot[column]) {
        let mut vector = vec![0_u64; words_for(ncols)];
        set_bit(&mut vector, free);
        for (row, &pivot) in reduced.iter().zip(&pivots) {
            if get_bit(row, free) {
                set_bit(&mut vector, pivot);
            }
        }
        basis.push(vector);
    }
    basis
}

fn satisfies_effect_groups(candidate: &BitRow, effect_groups: &[Vec<BitRow>]) -> bool {
    effect_groups
        .iter()
        .all(|group| group.iter().any(|observable| dot(candidate, observable)))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn trial_seed(namespace: u64, trial: usize) -> u64 {
    splitmix64(namespace ^ splitmix64(trial as u64))
}

fn random_location_group_order(
    location_groups: &LocationGroups,
    multi_qubit_group_order_weight: f64,
    multi_qubit_min_size: usize,
    preferred_group_mask: Option<&[bool]>,
    preferred_group_order_weight: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let mut groups: Vec<_> = (0..location_groups.len()).collect();
    if preferred_group_mask.is_some() {
        let preferred_group_mask = preferred_group_mask.expect("checked above");
        let mut keyed: Vec<_> = groups
            .drain(..)
            .map(|index| {
                let is_multi_qubit = location_groups[index].len() >= multi_qubit_min_size;
                let strict_multi_class =
                    usize::from(multi_qubit_group_order_weight == 0.0 && is_multi_qubit);
                let multi_qubit_weight = if is_multi_qubit && multi_qubit_group_order_weight > 0.0 {
                    multi_qubit_group_order_weight
                } else {
                    1.0
                };
                let preferred_weight = if preferred_group_mask[index] {
                    preferred_group_order_weight
                } else {
                    1.0
                };
                let uniform = 1.0 - rng.random::<f64>();
                (
                    strict_multi_class,
                    -uniform.ln() / (multi_qubit_weight * preferred_weight),
                    index,
                )
            })
            .collect();
        keyed.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        return keyed.into_iter().map(|(_, _, index)| index).collect();
    }
    if multi_qubit_group_order_weight == 0.0 {
        let (mut ordinary, mut multi_qubit): (Vec<_>, Vec<_>) = groups
            .drain(..)
            .partition(|&index| location_groups[index].len() < multi_qubit_min_size);
        ordinary.shuffle(rng);
        multi_qubit.shuffle(rng);
        ordinary.extend(multi_qubit);
        return ordinary;
    }
    if multi_qubit_group_order_weight == 1.0 {
        groups.shuffle(rng);
        return groups;
    }
    let mut keyed: Vec<_> = groups
        .drain(..)
        .map(|index| {
            let rate = if location_groups[index].len() >= multi_qubit_min_size {
                multi_qubit_group_order_weight
            } else {
                1.0
            };
            let uniform = 1.0 - rng.random::<f64>();
            (-uniform.ln() / rate, index)
        })
        .collect();
    keyed.sort_unstable_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    keyed.into_iter().map(|(_, index)| index).collect()
}

fn columns_for_location_group(
    group: &[usize],
    method: SearchMethod,
    qubits: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    if method == SearchMethod::BinaryGrouped {
        let mut columns = group.to_vec();
        columns.shuffle(rng);
        return columns;
    }
    let blocks = if matches!(
        method,
        SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal
    ) {
        3
    } else {
        2
    };
    let mut columns = Vec::with_capacity(blocks * group.len());
    for &qubit in group {
        for block in 0..blocks {
            columns.push(qubit + block * qubits);
        }
    }
    columns.shuffle(rng);
    columns
}

fn random_column_order(
    working_ncols: usize,
    method: SearchMethod,
    qubits: usize,
    rng: &mut ChaCha8Rng,
    location_groups: Option<&LocationGroups>,
    multi_qubit_group_order_weight: f64,
    preferred_group_mask: Option<&[bool]>,
    preferred_group_order_weight: f64,
) -> Vec<usize> {
    if let Some(groups) = location_groups {
        let multi_qubit_min_size = if method == SearchMethod::BinaryGrouped {
            3
        } else {
            2
        };
        let group_order = random_location_group_order(
            groups,
            multi_qubit_group_order_weight,
            multi_qubit_min_size,
            preferred_group_mask,
            preferred_group_order_weight,
            rng,
        );
        let mut columns = Vec::with_capacity(working_ncols);
        for group_index in group_order {
            columns.extend(columns_for_location_group(
                &groups[group_index],
                method,
                qubits,
                rng,
            ));
        }
        return columns;
    }
    match method {
        SearchMethod::Hamming => {
            let mut columns: Vec<_> = (0..working_ncols).collect();
            columns.shuffle(rng);
            columns
        }
        SearchMethod::Grouped
        | SearchMethod::BinaryGrouped
        | SearchMethod::LocalClifford
        | SearchMethod::LocationLinear => {
            let mut groups: Vec<_> = (0..qubits).collect();
            groups.shuffle(rng);
            let mut columns = Vec::with_capacity(working_ncols);
            for qubit in groups {
                if rng.random::<bool>() {
                    columns.push(qubit);
                    columns.push(qubit + qubits);
                } else {
                    columns.push(qubit + qubits);
                    columns.push(qubit);
                }
            }
            columns
        }
        SearchMethod::ThreeBlock => {
            let mut groups: Vec<_> = (0..qubits).collect();
            groups.shuffle(rng);
            let mut columns = Vec::with_capacity(working_ncols);
            for qubit in groups {
                let mut local = [qubit, qubit + qubits, qubit + 2 * qubits];
                local.shuffle(rng);
                columns.extend(local);
            }
            columns
        }
        SearchMethod::ThreeBlockGlobal => {
            let mut columns: Vec<_> = (0..working_ncols).collect();
            columns.shuffle(rng);
            columns
        }
    }
}

fn apply_local_map(x: bool, z: bool, map: u8) -> (bool, bool) {
    match map {
        0 => (x, z),
        1 => (z, x),
        2 => (x, x ^ z),
        3 => (x ^ z, z),
        4 => (z, x ^ z),
        5 => (x ^ z, x),
        _ => unreachable!("a local Clifford map is in 0..6"),
    }
}

fn inverse_local_map(map: u8) -> u8 {
    match map {
        0..=3 => map,
        4 => 5,
        5 => 4,
        _ => unreachable!("a local Clifford map is in 0..6"),
    }
}

fn transform_local_cliffords(row: &BitRow, maps: &[u8], inverse: bool) -> BitRow {
    let qubits = maps.len();
    let mut out = vec![0_u64; words_for(2 * qubits)];
    for (qubit, &forward_map) in maps.iter().enumerate() {
        let map = if inverse {
            inverse_local_map(forward_map)
        } else {
            forward_map
        };
        let (x, z) = apply_local_map(get_bit(row, qubit), get_bit(row, qubit + qubits), map);
        if x {
            set_bit(&mut out, qubit);
        }
        if z {
            set_bit(&mut out, qubit + qubits);
        }
    }
    out
}

#[derive(Clone, Debug)]
struct LocationLinearMap {
    columns: Vec<usize>,
    forward_rows: Vec<u8>,
    inverse_rows: Vec<u8>,
}

fn invert_small_binary_matrix(rows: &[u8], dimension: usize) -> Option<Vec<u8>> {
    let mut augmented: Vec<u16> = rows
        .iter()
        .enumerate()
        .map(|(index, &row)| u16::from(row) | (1_u16 << (dimension + index)))
        .collect();
    for column in 0..dimension {
        let pivot = (column..dimension).find(|&row| ((augmented[row] >> column) & 1) != 0)?;
        augmented.swap(column, pivot);
        let pivot_row = augmented[column];
        for (row, value) in augmented.iter_mut().enumerate() {
            if row != column && ((*value >> column) & 1) != 0 {
                *value ^= pivot_row;
            }
        }
    }
    let mask = (1_u16 << dimension) - 1;
    Some(
        augmented
            .into_iter()
            .map(|row| ((row >> dimension) & mask) as u8)
            .collect(),
    )
}

fn random_invertible_small_matrix(dimension: usize, rng: &mut ChaCha8Rng) -> (Vec<u8>, Vec<u8>) {
    debug_assert!((1..=4).contains(&dimension));
    loop {
        let rows: Vec<u8> = (0..dimension)
            .map(|_| rng.random_range(0..(1_u8 << dimension)))
            .collect();
        if let Some(inverse) = invert_small_binary_matrix(&rows, dimension) {
            return (rows, inverse);
        }
    }
}

fn apply_small_binary_matrix(value: u8, rows: &[u8]) -> u8 {
    rows.iter().enumerate().fold(0_u8, |result, (index, row)| {
        result | ((((value & row).count_ones() & 1) as u8) << index)
    })
}

fn random_location_linear_maps(
    groups: &LocationGroups,
    qubits: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<LocationLinearMap> {
    groups
        .iter()
        .map(|group| {
            let columns: Vec<_> = group
                .iter()
                .flat_map(|&qubit| [qubit, qubit + qubits])
                .collect();
            let (forward_rows, inverse_rows) = random_invertible_small_matrix(columns.len(), rng);
            LocationLinearMap {
                columns,
                forward_rows,
                inverse_rows,
            }
        })
        .collect()
}

fn random_binary_location_linear_maps(
    groups: &LocationGroups,
    rng: &mut ChaCha8Rng,
) -> Vec<LocationLinearMap> {
    groups
        .iter()
        .map(|group| {
            let (forward_rows, inverse_rows) = random_invertible_small_matrix(group.len(), rng);
            LocationLinearMap {
                columns: group.clone(),
                forward_rows,
                inverse_rows,
            }
        })
        .collect()
}

fn transform_location_linear(row: &BitRow, maps: &[LocationLinearMap], inverse: bool) -> BitRow {
    let mut out = vec![0_u64; row.len()];
    for map in maps {
        let value = map
            .columns
            .iter()
            .enumerate()
            .fold(0_u8, |value, (index, &column)| {
                value | (u8::from(get_bit(row, column)) << index)
            });
        let rows = if inverse {
            &map.inverse_rows
        } else {
            &map.forward_rows
        };
        let transformed = apply_small_binary_matrix(value, rows);
        for (index, &column) in map.columns.iter().enumerate() {
            if ((transformed >> index) & 1) != 0 {
                set_bit(&mut out, column);
            }
        }
    }
    out
}

fn embed_three_block(row: &BitRow, qubits: usize) -> BitRow {
    let mut out = vec![0_u64; words_for(3 * qubits)];
    for qubit in 0..qubits {
        let x = get_bit(row, qubit);
        let z = get_bit(row, qubit + qubits);
        if x {
            set_bit(&mut out, qubit);
        }
        if z {
            set_bit(&mut out, qubit + qubits);
        }
        if x ^ z {
            set_bit(&mut out, qubit + 2 * qubits);
        }
    }
    out
}

fn project_three_block(row: &BitRow, qubits: usize) -> BitRow {
    let mut out = vec![0_u64; words_for(2 * qubits)];
    for qubit in 0..qubits {
        let x = get_bit(row, qubit);
        let z = get_bit(row, qubit + qubits);
        debug_assert_eq!(get_bit(row, qubit + 2 * qubits), x ^ z);
        if x {
            set_bit(&mut out, qubit);
        }
        if z {
            set_bit(&mut out, qubit + qubits);
        }
    }
    out
}

fn evaluate_candidate(
    best: &mut TrialBest,
    candidate: &BitRow,
    effect_groups: &[Vec<BitRow>],
    method: SearchMethod,
    qubits: usize,
    location_groups: Option<&LocationGroups>,
    location_occupancies: Option<&LocationOccupancies>,
    scoring_cutoff: usize,
) {
    if !satisfies_effect_groups(candidate, effect_groups) {
        return;
    }
    let cutoff = best.distance.min(scoring_cutoff);
    let weight = if let Some(occupancies) = location_occupancies {
        let Some(weight) = bounded_location_weight(candidate, occupancies, cutoff) else {
            return;
        };
        weight
    } else {
        output_weight(candidate, method, qubits, location_groups)
    };
    best.consider(candidate, weight);
}

fn evaluate_working_candidate(
    best: &mut TrialBest,
    candidate: &BitRow,
    effect_groups: &[Vec<BitRow>],
    method: SearchMethod,
    qubits: usize,
    local_maps: &[u8],
    location_linear_maps: &[LocationLinearMap],
    binary_location_linear_maps: &[LocationLinearMap],
    location_groups: Option<&LocationGroups>,
    location_occupancies: Option<&LocationOccupancies>,
    scoring_cutoff: usize,
) {
    if method == SearchMethod::BinaryGrouped && !binary_location_linear_maps.is_empty() {
        let original = transform_location_linear(candidate, binary_location_linear_maps, true);
        evaluate_candidate(
            best,
            &original,
            effect_groups,
            method,
            qubits,
            location_groups,
            location_occupancies,
            scoring_cutoff,
        );
        return;
    }
    match method {
        SearchMethod::Hamming | SearchMethod::Grouped | SearchMethod::BinaryGrouped => {
            evaluate_candidate(
                best,
                candidate,
                effect_groups,
                method,
                qubits,
                location_groups,
                location_occupancies,
                scoring_cutoff,
            );
        }
        SearchMethod::LocalClifford => {
            let original = transform_local_cliffords(candidate, local_maps, true);
            evaluate_candidate(
                best,
                &original,
                effect_groups,
                method,
                qubits,
                location_groups,
                location_occupancies,
                scoring_cutoff,
            );
        }
        SearchMethod::LocationLinear => {
            let original = transform_location_linear(candidate, location_linear_maps, true);
            evaluate_candidate(
                best,
                &original,
                effect_groups,
                method,
                qubits,
                location_groups,
                location_occupancies,
                scoring_cutoff,
            );
        }
        SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal => {
            let original = project_three_block(candidate, qubits);
            evaluate_candidate(
                best,
                &original,
                effect_groups,
                method,
                qubits,
                location_groups,
                location_occupancies,
                scoring_cutoff,
            );
        }
    }
}

struct TrialWorkspace {
    reduced: Vec<BitRow>,
}

impl TrialWorkspace {
    fn new(kernel: &[BitRow]) -> Self {
        Self {
            reduced: kernel.to_vec(),
        }
    }

    fn reset(&mut self, kernel: &[BitRow]) {
        if self.reduced.len() != kernel.len() {
            self.reduced = kernel.to_vec();
            return;
        }
        for (target, source) in self.reduced.iter_mut().zip(kernel) {
            target.clone_from(source);
        }
    }
}

fn search_trial(
    kernel: &[BitRow],
    effect_groups: &[Vec<BitRow>],
    output_ncols: usize,
    method: SearchMethod,
    qubits: usize,
    seed: u64,
    pair_window: usize,
    location_groups: Option<&LocationGroups>,
    location_occupancies: Option<&LocationOccupancies>,
    multi_qubit_group_order_weight: f64,
    preferred_group_mask: Option<&[bool]>,
    preferred_group_order_weight: f64,
    randomize_within_location_basis: bool,
    scoring_cutoff: usize,
    workspace: Option<&mut TrialWorkspace>,
) -> TrialBest {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let local_maps: Vec<u8> = if method == SearchMethod::LocalClifford {
        (0..qubits).map(|_| rng.random_range(0..6)).collect()
    } else {
        vec![]
    };
    let location_linear_maps = if method == SearchMethod::LocationLinear {
        if let Some(groups) = location_groups {
            random_location_linear_maps(groups, qubits, &mut rng)
        } else {
            let groups = (0..qubits).map(|qubit| vec![qubit]).collect();
            random_location_linear_maps(&groups, qubits, &mut rng)
        }
    } else {
        vec![]
    };
    let binary_location_linear_maps =
        if method == SearchMethod::BinaryGrouped && randomize_within_location_basis {
            random_binary_location_linear_maps(
                location_groups.expect("binary basis randomization requires location groups"),
                &mut rng,
            )
        } else {
            vec![]
        };
    let mut owned_reduced: Vec<BitRow>;
    let reduced = if let Some(workspace) = workspace {
        debug_assert_eq!(method, SearchMethod::BinaryGrouped);
        debug_assert!(!randomize_within_location_basis);
        workspace.reset(kernel);
        &mut workspace.reduced
    } else {
        owned_reduced = match method {
            SearchMethod::LocalClifford => kernel
                .iter()
                .map(|row| transform_local_cliffords(row, &local_maps, false))
                .collect(),
            SearchMethod::LocationLinear => kernel
                .iter()
                .map(|row| transform_location_linear(row, &location_linear_maps, false))
                .collect(),
            SearchMethod::BinaryGrouped if randomize_within_location_basis => kernel
                .iter()
                .map(|row| transform_location_linear(row, &binary_location_linear_maps, false))
                .collect(),
            _ => kernel.to_vec(),
        };
        &mut owned_reduced
    };
    let working_ncols = match method {
        SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal => 3 * qubits,
        _ => output_ncols,
    };
    let order = random_column_order(
        working_ncols,
        method,
        qubits,
        &mut rng,
        location_groups,
        multi_qubit_group_order_weight,
        preferred_group_mask,
        preferred_group_order_weight,
    );
    let pivots = rref_with_order(reduced, &order);
    let mut best = TrialBest::empty(words_for(output_ncols));
    for row in reduced.iter() {
        evaluate_working_candidate(
            &mut best,
            row,
            effect_groups,
            method,
            qubits,
            &local_maps,
            &location_linear_maps,
            &binary_location_linear_maps,
            location_groups,
            location_occupancies,
            scoring_cutoff,
        );
    }

    if method != SearchMethod::Hamming {
        let mut pivot_rows = vec![None; working_ncols];
        for (row_index, &pivot) in pivots.iter().enumerate() {
            pivot_rows[pivot] = Some(row_index);
        }
        let local_columns = if method == SearchMethod::BinaryGrouped {
            1
        } else if matches!(
            method,
            SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal
        ) {
            3
        } else {
            2
        };
        let singleton_groups;
        let groups = if let Some(groups) = location_groups {
            groups
        } else {
            singleton_groups = (0..qubits).map(|qubit| vec![qubit]).collect();
            &singleton_groups
        };
        for group in groups {
            let mut rows = Vec::with_capacity(local_columns * group.len());
            for &coordinate in group {
                for block in 0..local_columns {
                    let column = if method == SearchMethod::BinaryGrouped {
                        coordinate
                    } else {
                        coordinate + block * qubits
                    };
                    if let Some(row_index) = pivot_rows[column] {
                        rows.push(row_index);
                    }
                }
            }
            for mask in 1_usize..(1_usize << rows.len()) {
                if mask.is_power_of_two() {
                    continue;
                }
                let mut candidate = vec![0_u64; reduced[0].len()];
                for (index, &row_index) in rows.iter().enumerate() {
                    if (mask >> index) & 1 != 0 {
                        xor_assign(&mut candidate, &reduced[row_index]);
                    }
                }
                evaluate_working_candidate(
                    &mut best,
                    &candidate,
                    effect_groups,
                    method,
                    qubits,
                    &local_maps,
                    &location_linear_maps,
                    &binary_location_linear_maps,
                    location_groups,
                    location_occupancies,
                    scoring_cutoff,
                );
            }
        }
    }

    if pair_window >= 2 {
        let mut ranked: Vec<_> = reduced
            .iter()
            .enumerate()
            .map(|(index, row)| {
                (
                    working_weight(row, method, qubits, location_groups, location_occupancies),
                    index,
                )
            })
            .collect();
        ranked.sort_unstable();
        ranked.truncate(pair_window.min(ranked.len()));
        for left in 0..ranked.len() {
            for right in (left + 1)..ranked.len() {
                let candidate = xor_rows(&reduced[ranked[left].1], &reduced[ranked[right].1]);
                evaluate_working_candidate(
                    &mut best,
                    &candidate,
                    effect_groups,
                    method,
                    qubits,
                    &local_maps,
                    &location_linear_maps,
                    &binary_location_linear_maps,
                    location_groups,
                    location_occupancies,
                    scoring_cutoff,
                );
            }
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn randomized_search(
    detectors: Vec<BitRow>,
    effect_groups: Vec<Vec<BitRow>>,
    ncols: usize,
    method: SearchMethod,
    trials: usize,
    seed: u64,
    min_distance: usize,
    threads: usize,
    batch_size: usize,
    pair_window: usize,
    supplied_location_groups: Option<Vec<Vec<usize>>>,
    multi_qubit_group_order_weight: f64,
    supplied_preferred_location_groups: Option<Vec<Vec<usize>>>,
    preferred_group_order_weight: f64,
    randomize_within_location_basis: bool,
) -> PyResult<SearchResult> {
    if trials == 0 {
        return Err(PyValueError::new_err("trials must be positive"));
    }
    if effect_groups.is_empty() || effect_groups.iter().any(Vec::is_empty) {
        return Err(PyValueError::new_err(
            "every required effect group must contain at least one row",
        ));
    }
    let started = Instant::now();
    let original_kernel = nullspace(&detectors, ncols);
    let qubits = if matches!(method, SearchMethod::Hamming | SearchMethod::BinaryGrouped) {
        ncols
    } else {
        ncols / 2
    };
    if !multi_qubit_group_order_weight.is_finite() || multi_qubit_group_order_weight < 0.0 {
        return Err(PyValueError::new_err(
            "multi_qubit_group_order_weight must be finite and nonnegative",
        ));
    }
    if supplied_location_groups.is_none() && multi_qubit_group_order_weight != 1.0 {
        return Err(PyValueError::new_err(
            "multi_qubit_group_order_weight requires location_groups",
        ));
    }
    if !preferred_group_order_weight.is_finite() || preferred_group_order_weight <= 0.0 {
        return Err(PyValueError::new_err(
            "preferred_group_order_weight must be finite and positive",
        ));
    }
    if supplied_preferred_location_groups.is_none() && preferred_group_order_weight != 1.0 {
        return Err(PyValueError::new_err(
            "preferred_group_order_weight requires preferred_location_groups",
        ));
    }
    if supplied_preferred_location_groups.is_some() && supplied_location_groups.is_none() {
        return Err(PyValueError::new_err(
            "preferred_location_groups requires location_groups",
        ));
    }
    if randomize_within_location_basis && method != SearchMethod::BinaryGrouped {
        return Err(PyValueError::new_err(
            "within-location basis randomization is only available for grouped binary search",
        ));
    }
    if randomize_within_location_basis && supplied_location_groups.is_none() {
        return Err(PyValueError::new_err(
            "within-location basis randomization requires location_groups",
        ));
    }
    let location_groups = if method == SearchMethod::BinaryGrouped {
        Some(normalize_binary_location_groups(
            supplied_location_groups,
            ncols,
        )?)
    } else {
        normalize_location_groups(supplied_location_groups, qubits)?
    };
    let location_occupancies = location_groups
        .as_ref()
        .map(|groups| build_location_occupancies(groups, qubits, method));
    let preferred_group_mask = match (location_groups.as_ref(), supplied_preferred_location_groups)
    {
        (_, None) => None,
        (Some(groups), Some(mut preferred)) => {
            let mut mask = vec![false; groups.len()];
            for group in &mut preferred {
                if group.is_empty() {
                    return Err(PyValueError::new_err(
                        "preferred location groups must be nonempty",
                    ));
                }
                group.sort_unstable();
                group.dedup();
                let Ok(index) = groups.binary_search(group) else {
                    return Err(PyValueError::new_err(format!(
                        "preferred location group {group:?} is not a complete location group",
                    )));
                };
                if std::mem::replace(&mut mask[index], true) {
                    return Err(PyValueError::new_err(format!(
                        "preferred location group {group:?} was supplied more than once",
                    )));
                }
            }
            Some(mask)
        }
        (None, Some(_)) => unreachable!("validated above"),
    };
    let kernel: Vec<BitRow> = if matches!(
        method,
        SearchMethod::ThreeBlock | SearchMethod::ThreeBlockGlobal
    ) {
        original_kernel
            .iter()
            .map(|row| embed_three_block(row, qubits))
            .collect()
    } else {
        original_kernel
    };
    let worker_count = if threads == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        threads
    };
    let batch = if batch_size == 0 {
        (4 * worker_count).max(1)
    } else {
        batch_size
    };
    let pool = ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut workspace = (worker_count == 1
        && method == SearchMethod::BinaryGrouped
        && !randomize_within_location_basis)
        .then(|| TrialWorkspace::new(&kernel));
    let mut completed = 0;
    let mut global = TrialBest::empty(words_for(ncols));
    while completed < trials {
        let end = (completed + batch).min(trials);
        let scoring_cutoff = global.distance;
        let partial = if worker_count == 1 {
            let mut partial = TrialBest::empty(words_for(ncols));
            for trial in completed..end {
                partial.merge(search_trial(
                    &kernel,
                    &effect_groups,
                    ncols,
                    method,
                    qubits,
                    trial_seed(seed, trial),
                    pair_window,
                    location_groups.as_ref(),
                    location_occupancies.as_ref(),
                    multi_qubit_group_order_weight,
                    preferred_group_mask.as_deref(),
                    preferred_group_order_weight,
                    randomize_within_location_basis,
                    scoring_cutoff,
                    workspace.as_mut(),
                ));
            }
            partial
        } else {
            pool.install(|| {
                (completed..end)
                    .into_par_iter()
                    .map(|trial| {
                        search_trial(
                            &kernel,
                            &effect_groups,
                            ncols,
                            method,
                            qubits,
                            trial_seed(seed, trial),
                            pair_window,
                            location_groups.as_ref(),
                            location_occupancies.as_ref(),
                            multi_qubit_group_order_weight,
                            preferred_group_mask.as_deref(),
                            preferred_group_order_weight,
                            randomize_within_location_basis,
                            scoring_cutoff,
                            None,
                        )
                    })
                    .reduce(
                        || TrialBest::empty(words_for(ncols)),
                        |mut left, right| {
                            left.merge(right);
                            left
                        },
                    )
            })
        };
        global.merge(partial);
        completed = end;
        if min_distance > 0 && global.distance <= min_distance {
            break;
        }
    }
    let distance = (global.distance != usize::MAX).then_some(global.distance);
    if distance.is_some() {
        let detector_zero = detectors
            .iter()
            .all(|detector| !dot(&global.witness, detector));
        let valid_effect = satisfies_effect_groups(&global.witness, &effect_groups);
        let exact_weight = location_occupancies.as_ref().map_or_else(
            || output_weight(&global.witness, method, qubits, location_groups.as_ref()),
            |occupancies| location_weight_with_occupancies(&global.witness, occupancies),
        );
        if !detector_zero || !valid_effect || Some(exact_weight) != distance {
            return Err(PyValueError::new_err(
                "internal error: returned witness failed exact validation",
            ));
        }
    }
    let witness = if distance.is_some() {
        (0..ncols)
            .map(|column| u8::from(get_bit(&global.witness, column)))
            .collect()
    } else {
        vec![]
    };
    Ok(SearchResult {
        distance,
        witness,
        trials_requested: trials,
        trials_completed: completed,
        hits: global.hits,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        target_hit: min_distance > 0 && distance.is_some_and(|value| value <= min_distance),
    })
}

#[pyfunction]
#[pyo3(signature = (checks, gauges, trials=10_000, seed=0, min_distance=0, threads=0, batch_size=0, pair_window=0))]
#[allow(clippy::too_many_arguments)]
fn search_css(
    checks: PyReadonlyArray2<'_, u8>,
    gauges: PyReadonlyArray2<'_, u8>,
    trials: usize,
    seed: u64,
    min_distance: usize,
    threads: usize,
    batch_size: usize,
    pair_window: usize,
) -> PyResult<SearchResult> {
    let (checks, ncols) = packed_rows(checks)?;
    let (gauges, gauge_ncols) = packed_rows(gauges)?;
    if ncols != gauge_ncols {
        return Err(PyValueError::new_err(
            "checks and gauges must have equal widths",
        ));
    }
    if checks
        .iter()
        .any(|check| gauges.iter().any(|gauge| dot(check, gauge)))
    {
        return Err(PyValueError::new_err(
            "checks and gauges are not orthogonal",
        ));
    }
    let gauge_dual = nullspace(&gauges, ncols);
    randomized_search(
        checks,
        vec![gauge_dual],
        ncols,
        SearchMethod::Hamming,
        trials,
        seed,
        min_distance,
        threads,
        batch_size,
        pair_window,
        None,
        1.0,
        None,
        1.0,
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (detectors, observables, trials=10_000, seed=0, min_distance=0, threads=0, batch_size=0, pair_window=0, method="grouped", location_groups=None, multi_qubit_group_order_weight=1.0, preferred_location_groups=None, preferred_group_order_weight=1.0))]
#[allow(clippy::too_many_arguments)]
fn search_symplectic(
    detectors: PyReadonlyArray2<'_, u8>,
    observables: PyReadonlyArray2<'_, u8>,
    trials: usize,
    seed: u64,
    min_distance: usize,
    threads: usize,
    batch_size: usize,
    pair_window: usize,
    method: &str,
    location_groups: Option<Vec<Vec<usize>>>,
    multi_qubit_group_order_weight: f64,
    preferred_location_groups: Option<Vec<Vec<usize>>>,
    preferred_group_order_weight: f64,
) -> PyResult<SearchResult> {
    let (detectors, ncols) = packed_rows(detectors)?;
    let (observables, observable_ncols) = packed_rows(observables)?;
    if ncols != observable_ncols {
        return Err(PyValueError::new_err(
            "detectors and observables must have equal widths",
        ));
    }
    if ncols % 2 != 0 {
        return Err(PyValueError::new_err(
            "symplectic matrices must have width 2n",
        ));
    }
    randomized_search(
        detectors,
        vec![observables],
        ncols,
        SearchMethod::parse(method)?,
        trials,
        seed,
        min_distance,
        threads,
        batch_size,
        pair_window,
        location_groups,
        multi_qubit_group_order_weight,
        preferred_location_groups,
        preferred_group_order_weight,
        false,
    )
}

/// Search an arbitrary binary detector/observable problem with grouped-column cost.
///
/// Explicit location groups contain one to four original matrix columns. Columns
/// omitted from all explicit groups become singleton locations. For ordering bias,
/// groups of size three or four are the multi-qubit class; groups of size one or
/// two are the baseline class. The returned witness uses the original columns.
#[pyfunction]
#[pyo3(signature = (detectors, observables, trials=10_000, seed=0, min_distance=0, threads=0, batch_size=0, pair_window=0, location_groups=None, multi_qubit_group_order_weight=1.0, preferred_location_groups=None, preferred_group_order_weight=1.0, randomize_within_location_basis=false))]
#[allow(clippy::too_many_arguments)]
fn search_grouped_binary(
    detectors: PyReadonlyArray2<'_, u8>,
    observables: PyReadonlyArray2<'_, u8>,
    trials: usize,
    seed: u64,
    min_distance: usize,
    threads: usize,
    batch_size: usize,
    pair_window: usize,
    location_groups: Option<Vec<Vec<usize>>>,
    multi_qubit_group_order_weight: f64,
    preferred_location_groups: Option<Vec<Vec<usize>>>,
    preferred_group_order_weight: f64,
    randomize_within_location_basis: bool,
) -> PyResult<SearchResult> {
    let (detectors, ncols) = packed_rows(detectors)?;
    let (observables, observable_ncols) = packed_rows(observables)?;
    if ncols != observable_ncols {
        return Err(PyValueError::new_err(
            "detectors and observables must have equal widths",
        ));
    }
    randomized_search(
        detectors,
        vec![observables],
        ncols,
        SearchMethod::BinaryGrouped,
        trials,
        seed,
        min_distance,
        threads,
        batch_size,
        pair_window,
        location_groups,
        multi_qubit_group_order_weight,
        preferred_location_groups,
        preferred_group_order_weight,
        randomize_within_location_basis,
    )
}

#[pyfunction]
#[pyo3(signature = (detectors, observables, required_effects, trials=10_000, seed=0, min_distance=0, threads=0, batch_size=0, pair_window=0, method="grouped", location_groups=None, multi_qubit_group_order_weight=1.0, preferred_location_groups=None, preferred_group_order_weight=1.0))]
#[allow(clippy::too_many_arguments)]
fn search_symplectic_constrained(
    detectors: PyReadonlyArray2<'_, u8>,
    observables: PyReadonlyArray2<'_, u8>,
    required_effects: PyReadonlyArray2<'_, u8>,
    trials: usize,
    seed: u64,
    min_distance: usize,
    threads: usize,
    batch_size: usize,
    pair_window: usize,
    method: &str,
    location_groups: Option<Vec<Vec<usize>>>,
    multi_qubit_group_order_weight: f64,
    preferred_location_groups: Option<Vec<Vec<usize>>>,
    preferred_group_order_weight: f64,
) -> PyResult<SearchResult> {
    let (detectors, ncols) = packed_rows(detectors)?;
    let (observables, observable_ncols) = packed_rows(observables)?;
    let (required_effects, required_ncols) = packed_rows(required_effects)?;
    if ncols != observable_ncols || ncols != required_ncols {
        return Err(PyValueError::new_err(
            "detectors, observables, and required_effects must have equal widths",
        ));
    }
    if ncols % 2 != 0 {
        return Err(PyValueError::new_err(
            "symplectic matrices must have width 2n",
        ));
    }
    randomized_search(
        detectors,
        vec![observables, required_effects],
        ncols,
        SearchMethod::parse(method)?,
        trials,
        seed,
        min_distance,
        threads,
        batch_size,
        pair_window,
        location_groups,
        multi_qubit_group_order_weight,
        preferred_location_groups,
        preferred_group_order_weight,
        false,
    )
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SearchResult>()?;
    module.add_function(wrap_pyfunction!(search_css, module)?)?;
    module.add_function(wrap_pyfunction!(search_grouped_binary, module)?)?;
    module.add_function(wrap_pyfunction!(search_symplectic, module)?)?;
    module.add_function(wrap_pyfunction!(search_symplectic_constrained, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(bits: &[usize], ncols: usize) -> BitRow {
        let mut out = vec![0; words_for(ncols)];
        for &bit in bits {
            set_bit(&mut out, bit);
        }
        out
    }

    #[test]
    fn nullspace_vectors_satisfy_checks() {
        let checks = vec![row(&[0, 1, 2], 4), row(&[1, 2, 3], 4)];
        let kernel = nullspace(&checks, 4);
        assert_eq!(kernel.len(), 2);
        assert!(
            kernel
                .iter()
                .all(|vector| checks.iter().all(|check| !dot(vector, check)))
        );
    }

    #[test]
    fn symplectic_weight_counts_y_once() {
        let value = row(&[0, 1, 3, 5], 6);
        assert_eq!(symplectic_weight(&value, 3), 3);
    }

    #[test]
    fn location_weight_counts_a_two_qubit_group_once() {
        let value = row(&[0, 4], 6);
        let groups = vec![vec![0, 1], vec![2]];
        assert_eq!(symplectic_weight(&value, 3), 2);
        assert_eq!(location_weight(&value, 3, &groups), 1);
    }

    #[test]
    fn packed_location_occupancies_match_reference_across_words() {
        let qubits = 70;
        let groups = vec![vec![0], vec![1, 64], vec![2, 69], vec![3]];
        let occupancies = build_location_occupancies(&groups, qubits, SearchMethod::Grouped);
        for bits in [
            vec![],
            vec![0],
            vec![64],
            vec![1 + qubits],
            vec![2, 69 + qubits],
            vec![3, 64, 69 + qubits],
        ] {
            let value = row(&bits, 2 * qubits);
            assert_eq!(
                location_weight_with_occupancies(&value, &occupancies),
                location_weight(&value, qubits, &groups)
            );
        }
    }

    #[test]
    fn bounded_location_weight_keeps_ties_and_stops_only_above_cutoff() {
        let qubits = 4;
        let groups = vec![vec![0, 1], vec![2], vec![3]];
        let occupancies = build_location_occupancies(&groups, qubits, SearchMethod::Grouped);
        let value = row(&[0, 2, 7], 2 * qubits);
        assert_eq!(location_weight(&value, qubits, &groups), 3);
        assert_eq!(bounded_location_weight(&value, &occupancies, 3), Some(3));
        assert_eq!(bounded_location_weight(&value, &occupancies, 2), None);
        assert_eq!(
            bounded_location_weight(&value, &occupancies, usize::MAX),
            Some(3)
        );
    }

    #[test]
    fn partial_location_groups_gain_singleton_complements() {
        let groups = normalize_location_groups(Some(vec![vec![3, 1]]), 5)
            .unwrap()
            .unwrap();
        assert_eq!(groups, vec![vec![0], vec![1, 3], vec![2], vec![4]]);
    }

    #[test]
    fn exponential_order_weight_moves_multi_qubit_groups_earlier() {
        let mut groups = vec![vec![0, 1]];
        groups.extend((2..20).map(|qubit| vec![qubit]));
        let mut unbiased_rank_sum = 0;
        let mut biased_rank_sum = 0;
        for seed in 0..2_000 {
            let mut unbiased_rng = ChaCha8Rng::seed_from_u64(seed);
            let mut biased_rng = ChaCha8Rng::seed_from_u64(seed);
            let unbiased =
                random_location_group_order(&groups, 1.0, 2, None, 1.0, &mut unbiased_rng);
            let biased = random_location_group_order(&groups, 4.0, 2, None, 1.0, &mut biased_rng);
            unbiased_rank_sum += unbiased.iter().position(|&index| index == 0).unwrap();
            biased_rank_sum += biased.iter().position(|&index| index == 0).unwrap();
        }
        assert!(2 * biased_rank_sum < unbiased_rank_sum);
    }

    #[test]
    fn zero_order_weight_places_multi_qubit_groups_last() {
        let groups = vec![vec![0, 1], vec![2], vec![3, 4], vec![5]];
        for seed in 0..100 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let order = random_location_group_order(&groups, 0.0, 2, None, 1.0, &mut rng);
            let first_multi = order
                .iter()
                .position(|&index| groups[index].len() > 1)
                .unwrap();
            assert_eq!(first_multi, 2);
            assert!(
                order[..first_multi]
                    .iter()
                    .all(|&index| groups[index].len() == 1)
            );
            assert!(
                order[first_multi..]
                    .iter()
                    .all(|&index| groups[index].len() > 1)
            );
        }
    }

    #[test]
    fn binary_order_bias_treats_only_rank_three_and_four_as_multi_qubit() {
        let groups = vec![vec![0], vec![1, 2], vec![3, 4, 5], vec![6, 7, 8, 9]];
        for seed in 0..100 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let order = random_location_group_order(&groups, 0.0, 3, None, 1.0, &mut rng);
            assert!(order[..2].iter().all(|&index| groups[index].len() <= 2));
            assert!(order[2..].iter().all(|&index| groups[index].len() >= 3));
        }
    }

    #[test]
    fn preferred_order_weight_moves_selected_groups_later() {
        let groups: Vec<_> = (0..20).map(|column| vec![column]).collect();
        let mut preferred = vec![false; groups.len()];
        preferred[7] = true;
        let mut neutral_rank_sum = 0;
        let mut preferred_rank_sum = 0;
        for seed in 0..2_000 {
            let mut neutral_rng = ChaCha8Rng::seed_from_u64(seed);
            let mut preferred_rng = ChaCha8Rng::seed_from_u64(seed);
            let neutral = random_location_group_order(&groups, 1.0, 2, None, 1.0, &mut neutral_rng);
            let delayed = random_location_group_order(
                &groups,
                1.0,
                2,
                Some(&preferred),
                0.05,
                &mut preferred_rng,
            );
            neutral_rank_sum += neutral.iter().position(|&index| index == 7).unwrap();
            preferred_rank_sum += delayed.iter().position(|&index| index == 7).unwrap();
        }
        assert!(preferred_rank_sum > 3 * neutral_rank_sum / 2);
    }

    #[test]
    fn adjacent_namespaces_do_not_reorder_the_same_trial_keys() {
        use std::collections::HashSet;

        let first: HashSet<_> = (0..4_096).map(|trial| trial_seed(0, trial)).collect();
        let second: HashSet<_> = (0..4_096).map(|trial| trial_seed(1, trial)).collect();
        assert!(first.is_disjoint(&second));
    }

    #[test]
    fn local_clifford_maps_are_invertible_and_preserve_pairing() {
        for map in 0..6 {
            for left in 0..4 {
                let left_pair = (left & 1 != 0, left & 2 != 0);
                let transformed_left = apply_local_map(left_pair.0, left_pair.1, map);
                let recovered = apply_local_map(
                    transformed_left.0,
                    transformed_left.1,
                    inverse_local_map(map),
                );
                assert_eq!(recovered, left_pair);

                for right in 0..4 {
                    let right_pair = (right & 1 != 0, right & 2 != 0);
                    let transformed_right = apply_local_map(right_pair.0, right_pair.1, map);
                    assert_eq!(
                        (left_pair.0 & right_pair.1) ^ (left_pair.1 & right_pair.0),
                        (transformed_left.0 & transformed_right.1)
                            ^ (transformed_left.1 & transformed_right.0)
                    );
                }
            }
        }
    }

    #[test]
    fn small_binary_matrices_round_trip_every_vector() {
        let mut rng = ChaCha8Rng::seed_from_u64(91);
        for dimension in 1..=4 {
            for _ in 0..100 {
                let (forward, inverse) = random_invertible_small_matrix(dimension, &mut rng);
                for value in 0..(1_u8 << dimension) {
                    let transformed = apply_small_binary_matrix(value, &forward);
                    let recovered = apply_small_binary_matrix(transformed, &inverse);
                    assert_eq!(recovered, value);
                }
            }
        }
    }

    #[test]
    fn location_linear_maps_round_trip_and_preserve_group_cost() {
        let groups = vec![vec![0, 2], vec![1], vec![3, 4]];
        let qubits = 5;
        let mut rng = ChaCha8Rng::seed_from_u64(903);
        for _ in 0..100 {
            let maps = random_location_linear_maps(&groups, qubits, &mut rng);
            let original = row(&[0, 2, 6, 9], 2 * qubits);
            let transformed = transform_location_linear(&original, &maps, false);
            let recovered = transform_location_linear(&transformed, &maps, true);
            assert_eq!(recovered, original);
            assert_eq!(
                location_weight(&transformed, qubits, &groups),
                location_weight(&original, qubits, &groups)
            );
        }
    }

    #[test]
    fn three_block_embedding_round_trips_and_doubles_weight() {
        let original = row(&[0, 2, 4, 5], 6);
        let embedded = embed_three_block(&original, 3);
        assert_eq!(
            hamming_weight(&embedded),
            2 * symplectic_weight(&original, 3)
        );
        assert_eq!(project_three_block(&embedded, 3), original);
    }
}
