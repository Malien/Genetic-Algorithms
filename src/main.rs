#![feature(iter_array_chunks)]
#![feature(generic_const_exprs)]
use std::{
    fmt::Write,
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::atomic::AtomicUsize,
};

use bitvec::{order::Lsb0, slice::BitSlice, BitArr};
use decorum::{Finite, N64, R64};
use num_traits::{real::Real, FromPrimitive};
use operators::{crossover, mutation};
use rand::rngs::StdRng;

mod evaluation;
mod operators;
mod selection;
mod stats;

use evaluation::{
    binary_to_gray, evaluate, optimal_binary_specimen, optimal_pheonotype_specimen, BinaryAlgo,
    EvaluateFamily, GenomeEncoding, PheonotypeAlgo,
};
use selection::{selection, Selection, SelectionResult, TournamentReplacement};
use stats::{RunStats, StatEncoder, SuccessFamily};

use crate::stats::{ConfigStatsWithOptimum, ConfigStats};

const MAX_GENERATIONS: usize = 10_000_000;
const MAX_RUNS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RunKey<T> {
    algo_type: T,
    population_size: usize,
    apply_crossover: bool,
    apply_mutation: bool,
    selection: Selection,
    run_idx: usize,
}

impl<T: std::fmt::Display> RunKey<T> {
    fn to_path(&self) -> PathBuf {
        let mut path = PathBuf::new();

        let mut algo = String::new();
        write!(algo, "{}", self.algo_type).unwrap();
        if self.apply_crossover {
            write!(algo, "-crossover").unwrap();
        }
        if self.apply_mutation {
            write!(algo, "-mutation").unwrap();
        }

        path.push(algo);
        path.push(format!("{}", self.population_size));
        path.push(format!("{}", self.selection));
        path.push(format!("{}", self.run_idx));
        path
    }
}

#[derive(Debug, Clone)]
pub struct AlgoConfig<F: GenFamily + ?Sized>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    ty: F::AlgoType,
    population_size: usize,
    apply_crossover: bool,
    mutation_rate: Option<f64>,
    selection: Selection,
    optimal_specimen: Option<Genome<{ F::N }>>,
}

pub struct RunState<'a, F: GenFamily + ?Sized>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    config: &'a AlgoConfig<F>,
    rng: StdRng,
    run_idx: usize,
    stats: StatEncoder<'a, F>,
}

impl<'a, F: FullFamily> RunState<'a, F>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    fn new(config: &'a AlgoConfig<F>, run_idx: usize) -> Self {
        Self {
            config,
            rng: rand::SeedableRng::seed_from_u64(run_idx as u64),
            stats: StatEncoder::new(config),
            run_idx,
        }
    }

    fn initial_population(&mut self) -> Vec<Genome<{ F::N }>> {
        match self.config.optimal_specimen.clone() {
            Some(optimal_specimen) => (1..self.config.population_size)
                .map(|_| random_genome::<{ F::N }>(&mut self.rng))
                .chain(Some(optimal_specimen))
                .collect(),
            None => (0..self.config.population_size)
                .map(|_| random_genome::<{ F::N }>(&mut self.rng))
                .collect(),
        }
    }

    fn record_run_stat(
        &mut self,
        pre_selection_fitness: N64,
        population: &[EvaluatedGenome<{ F::N }>],
        unique_specimens_selected: usize,
    ) -> N64 {
        self.stats
            .record_run_stat(pre_selection_fitness, population, unique_specimens_selected)
    }

    fn finish_converged(
        self,
        final_population: &[EvaluatedGenome<{ F::N }>],
    ) -> (RunKey<F::AlgoType>, RunStats) {
        let key = RunKey {
            algo_type: self.config.ty,
            population_size: self.config.population_size,
            apply_crossover: self.config.apply_crossover,
            apply_mutation: self.config.mutation_rate.is_some(),
            selection: self.config.selection,
            run_idx: self.run_idx,
        };

        let stats = self.stats.finish_converged(final_population);
        (key, stats)
    }

    fn finish_unconverged(
        self,
        final_population: &[EvaluatedGenome<{ F::N }>],
    ) -> (RunKey<F::AlgoType>, RunStats) {
        let key = RunKey {
            algo_type: self.config.ty,
            population_size: self.config.population_size,
            apply_crossover: self.config.apply_crossover,
            apply_mutation: self.config.mutation_rate.is_some(),
            selection: self.config.selection,
            run_idx: self.run_idx,
        };

        let stats = self.stats.finish_unconverged(final_population);
        (key, stats)
    }
}

// I don't think we should disambiguate between genome lengths on the type level for now.
// I think the biggest win would be from removing allocations.
//
// Disabmiguation on the type level may help, since it would allow to pack smaller genomes more
// efficiently into a cache line.
//
// The idea would be to store a 128 bit wide array, and a usize to indicate the length.
// Afterwards just implement deref to the BitSlice of the desired size.
// This is basiacally what ArrayVec crate does, but for bitvecs
//
// Alternatively, we may pack the 120 bit wide array, and an u8 size. This may improve struct
// aligmnet. (Or if the struct is not packed, we may use a 112 bit wide array, and a u16 size)
// Either way, the fine-grained research is not worth it, since I am striving towards type level
// disambiguation in the end.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Genome<const N: usize>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    arr: BitArr![for N, in u16, Lsb0],
}

impl<const N: usize> Genome<N>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    fn into_inner(self) -> [u16; bitvec::mem::elts::<u16>(N)] {
        self.arr.into_inner()
    }

    fn new(arr: BitArr![for N, in u16, Lsb0]) -> Self {
        Self { arr }
    }
}

impl<const N: usize> Deref for Genome<N>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    type Target = BitSlice<u16, Lsb0>;

    fn deref(&self) -> &Self::Target {
        &self.arr[..N]
    }
}

impl<const N: usize> IntoIterator for Genome<N>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    type Item = bool;
    type IntoIter =
        std::iter::Take<bitvec::array::IntoIter<[u16; bitvec::mem::elts::<u16>(N)], Lsb0>>;

    fn into_iter(self) -> Self::IntoIter {
        self.arr.into_iter().take(N)
    }
}

impl<const N: usize> DerefMut for Genome<N>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arr[..N]
    }
}

pub trait GenFamily {
    const N: usize;
    type AlgoType: Copy;
}
pub struct G10;
pub struct G100;

pub trait FullFamily: GenFamily + EvaluateFamily + SuccessFamily {}
impl<T> FullFamily for T where T: GenFamily + EvaluateFamily + SuccessFamily {}

impl GenFamily for G10 {
    const N: usize = 10;
    type AlgoType = (PheonotypeAlgo, GenomeEncoding);
}

impl GenFamily for G100 {
    const N: usize = 100;
    type AlgoType = BinaryAlgo;
}

#[derive(Debug, Clone, Copy)]
pub struct EvaluatedGenome<const LEN: usize>
where
    [(); bitvec::mem::elts::<u16>(LEN)]:,
{
    genome: Genome<LEN>,
    fitness: N64,
}

pub fn random_genome<const N: usize>(rng: &mut impl rand::Rng) -> Genome<N>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    let mut genome = Genome::default();
    for idx in 0..N {
        genome.set(idx, rng.gen());
    }
    genome
}

fn homougeneousness<'a, const N: usize>(population: impl ExactSizeIterator<Item = Genome<N>>) -> R64
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    let len = population.len();
    let mut bit_sum = [Finite::from(0.0); N];
    for genome in population {
        for (i, bit) in genome.iter().enumerate() {
            if *bit {
                bit_sum[i] += 1.0;
            }
        }
    }
    for bit_count in &mut bit_sum {
        *bit_count =
            Finite::abs(*bit_count / (Finite::from_usize(len).unwrap()) - Finite::from(0.5))
                + Finite::from(0.5);
    }

    bit_sum
        .into_iter()
        .min()
        .expect("Population not to be empty")
}

#[test]
fn same_genomes_are_homougeneous() {
    use bitvec::{bitarr, order::Lsb0};

    let population = vec![
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
    ];
    assert_eq!(
        homougeneousness::<10>(population.into_iter().map(Genome::new)),
        1.0
    );
}

#[test]
fn almost_identical_gemoes_are_somewhat_homougeneous() {
    use bitvec::{bitarr, order::Lsb0};

    let population = vec![
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 1, 0, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 1, 0, 0, 1, 1, 0, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 0, 0, 1],
        bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 0],
    ];
    assert_eq!(
        homougeneousness::<10>(population.into_iter().map(Genome::new)),
        0.8
    );
}

const HOMOUGENEOUSNESS_THRESHOLD: f64 = 0.99;

fn avg_fitness<const N: usize>(population: &[EvaluatedGenome<N>]) -> N64
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    population.iter().map(|g| g.fitness).sum::<N64>() / N64::from_usize(population.len()).unwrap()
}

fn simulation<F: FullFamily>(mut state: RunState<F>) -> (RunKey<F::AlgoType>, RunStats)
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    let starting_population = state.initial_population();
    let mut starting_population = evaluate(&mut state, starting_population);
    let mut avg_fitness = avg_fitness(&starting_population);
    for _ in 0..MAX_GENERATIONS {
        let SelectionResult {
            new_population: population,
            unique_specimens_selected,
        } = selection(&mut state, starting_population);
        avg_fitness = state.record_run_stat(avg_fitness, &population, unique_specimens_selected);
        if is_convergant(&state, &population) {
            return state.finish_converged(&population);
        }
        let population = crossover(&mut state, population);
        let population = mutation(&mut state, population);
        let population = evaluate(&mut state, population);
        starting_population = population;
    }
    return state.finish_unconverged(&starting_population);
}

fn is_convergant<F: FullFamily>(
    state: &RunState<F>,
    population: &[EvaluatedGenome<{ F::N }>],
) -> bool
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    if state.config.mutation_rate.is_none() {
        return is_all_same(population.iter().map(|g| g.genome));
    }
    return homougeneousness::<{ F::N }>(population.iter().map(|g| g.genome))
        > HOMOUGENEOUSNESS_THRESHOLD;
}

fn is_all_same(iter: impl IntoIterator<Item = impl Eq>) -> bool {
    let mut iter = iter.into_iter();
    if let Some(first) = iter.next() {
        iter.all(|x| x == first)
    } else {
        true
    }
}

macro_rules! perms {
    ($pat:pat in $values:expr; $body:block) => {
        for $pat in $values {
            $body
        }
    };
    ($pat:pat in $values:expr; $($rest:tt)*) => {
        perms!($pat in $values; { perms!($($rest)*) })
    };
}

fn config_permutations_100() -> Vec<AlgoConfig<G100>> {
    let mut res = vec![];

    perms! {
        (population_size, mutation_rate) in [(100, 0.00001)];
        apply_crossover in [true, false];
        apply_mutation in [true, false];
        selection_prob in [1.0, 0.8, 0.7, 0.6];
        replacement in [TournamentReplacement::With, TournamentReplacement::Without];
        algo in [BinaryAlgo::FConst, BinaryAlgo::FHD { sigma: 100.0.into() }];
        {
            res.push(AlgoConfig {
                ty: algo,
                population_size,
                apply_crossover,
                mutation_rate: if apply_mutation { Some(mutation_rate) } else { None },
                selection: Selection::StochasticTournament {
                    prob: selection_prob.into(),
                    replacement,
                },
                optimal_specimen: optimal_binary_specimen(algo),
            });
        }
    };

    res
}

fn config_permutations_10() -> Vec<AlgoConfig<G10>> {
    let mut res = vec![];

    perms! {
        (population_size, mutation_rate) in [(100, 0.0005)];
        apply_crossover in [true, false];
        apply_mutation in [true, false];
        selection_prob in [1.0, 0.8, 0.7, 0.6];
        replacement in [TournamentReplacement::With, TournamentReplacement::Without];
        algo in [PheonotypeAlgo::Pow1, PheonotypeAlgo::Pow2];
        (encoding, optimal_specimen) in [
            (GenomeEncoding::Binary, optimal_pheonotype_specimen(algo)),
            (GenomeEncoding::BinaryGray, binary_to_gray(optimal_pheonotype_specimen(algo)))
        ];
        {
            res.push(AlgoConfig {
                ty: (algo, encoding),
                population_size,
                apply_crossover,
                mutation_rate: if apply_mutation { Some(mutation_rate) } else { None },
                selection: Selection::StochasticTournament {
                    prob: selection_prob.into(),
                    replacement,
                },
                optimal_specimen: Some(optimal_specimen),
            });
        }
    }

    res
}

fn main() {
    let config_10 = config_permutations_10();
    let config_100 = config_permutations_100();
    let len = config_10.len() + config_100.len();
    println!("Running {} configs", len);
    let counter = AtomicUsize::new(0);

    #[cfg(feature = "parallel")]
    rayon::scope(|s| {
        for (i, config) in config_10.into_iter().enumerate() {
            let counter = &counter;
            s.spawn(move |_| {
                // println!("Running config: {:#?}", config);
                let solved = counter.load(std::sync::atomic::Ordering::SeqCst);
                println!(
                    "Solved {}/{} ({}%)\tConfiguration #{i}: {}-{}/crossover={}/mutation={}/{}",
                    solved + 1,
                    len,
                    (solved + 1) * 100 / len,
                    config.ty.0,
                    config.ty.1,
                    config.apply_crossover,
                    config.mutation_rate.is_some(),
                    config.selection,
                );
                let runs = (0..MAX_RUNS)
                    .map(|run_idx| {
                        let state = RunState::new(&config, run_idx);
                        let (_key, stats) = simulation(state);
                        stats
                    })
                    .collect();
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _stats = ConfigStats::new(runs);
            });
        }
        for (i, config) in config_100.into_iter().enumerate() {
            let counter = &counter;
            s.spawn(move |_| {
                // println!("Running config: {:#?}", config);
                let solved = counter.load(std::sync::atomic::Ordering::SeqCst);
                println!(
                    "Solved {}/{} ({}%)\tConfiguration #{i}: {}/crossover={}/mutation={}/{}",
                    solved + 1,
                    len,
                    (solved + 1) * 100 / len,
                    config.ty,
                    config.apply_crossover,
                    config.mutation_rate.is_some(),
                    config.selection,
                );
                let runs = (0..MAX_RUNS)
                    .map(|run_idx| {
                        let state = RunState::new(&config, run_idx);
                        let (_key, stats) = simulation(state);
                        stats
                    })
                    .collect();
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _stats = ConfigStats::new(runs);
            });
        }
    });

    #[cfg(not(feature = "parallel"))]
    for (i, config) in config_10.into_iter().enumerate() {
        let counter = &counter;
        // println!("Running config: {:#?}", config);
        let solved = counter.load(std::sync::atomic::Ordering::SeqCst);
        println!(
            "Solved {}/{} ({}%)\tConfiguration #{i}: {}-{}/crossover={}/mutation={}/{}",
            solved + 1,
            len,
            (solved + 1) * 100 / len,
            config.ty.0,
            config.ty.1,
            config.apply_crossover,
            config.mutation_rate.is_some(),
            config.selection,
        );
        let runs = (0..MAX_RUNS)
            .map(|run_idx| {
                let state = RunState::new(&config, run_idx);
                let (_key, stats) = simulation(state);
                stats
            })
            .collect();
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _stats = ConfigStats::new(runs);
    }
    #[cfg(not(feature = "parallel"))]
    for (i, config) in config_100.into_iter().enumerate() {
        let counter = &counter;
        // println!("Running config: {:#?}", config);
        let solved = counter.load(std::sync::atomic::Ordering::SeqCst);
        println!(
            "Solved {}/{} ({}%)\tConfiguration #{i}: {}/crossover={}/mutation={}/{}",
            solved + 1,
            len,
            (solved + 1) * 100 / len,
            config.ty,
            config.apply_crossover,
            config.mutation_rate.is_some(),
            config.selection,
        );
        let runs = (0..MAX_RUNS)
            .map(|run_idx| {
                let state = RunState::new(&config, run_idx);
                let (_key, stats) = simulation(state);
                stats
            })
            .collect();
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _stats = ConfigStats::new(runs);
    }
}
