#![feature(iter_array_chunks)]
#![feature(generic_const_exprs)]
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::{fmt::Write, path::PathBuf, sync::atomic::AtomicUsize};

use bitvec::bitarr;
use bitvec::prelude::BitArray;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use bitvec::{order::Lsb0, BitArr};
use decorum::{Finite, NotNan, N64, R64};
use num_traits::{real::Real, FromPrimitive};
use operators::{crossover, mutation};
use rand::rngs::StdRng;

mod evaluation;
mod operators;
mod selection;
mod stats;

use evaluation::{
    binary_to_gray, evaluate, optimal_binary_specimen, optimal_pheonotype_specimen, pow1, AlgoType,
    BinaryAlgo, GenomeEncoding, PheonotypeAlgo,
};
use selection::{selection, Selection, SelectionResult, TournamentReplacement};
use stats::{RunStats, StatEncoder};

const MAX_GENERATIONS: usize = 10_000_000;
const MAX_RUNS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RunKey {
    algo_type: AlgoType,
    population_size: usize,
    apply_crossover: bool,
    apply_mutation: bool,
    selection: Selection,
    run_idx: usize,
}

impl RunKey {
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
pub struct AlgoConfig {
    ty: AlgoType,
    population_size: usize,
    apply_crossover: bool,
    mutation_rate: Option<f64>,
    selection: Selection,
    genome_len: GenomeLength,
    optimal_specimen: Option<Genome>,
}

pub struct RunState<'a> {
    config: &'a AlgoConfig,
    rng: StdRng,
    run_idx: usize,
    stats: StatEncoder<'a>,
}

impl<'a> RunState<'a> {
    fn new(config: &'a AlgoConfig, run_idx: usize) -> Self {
        Self {
            config,
            rng: rand::SeedableRng::seed_from_u64(run_idx as u64),
            stats: StatEncoder::new(config),
            run_idx,
        }
    }

    fn initial_population(&mut self) -> Vec<Genome> {
        match self.config.optimal_specimen.clone() {
            Some(optimal_specimen) => (1..self.config.population_size)
                .map(|_| random_genome(&mut self.rng, self.config.genome_len))
                .chain(Some(optimal_specimen))
                .collect(),
            None => (0..self.config.population_size)
                .map(|_| random_genome(&mut self.rng, self.config.genome_len))
                .collect(),
        }
    }

    fn record_run_stat(
        &mut self,
        pre_selection_fitness: N64,
        population: &[EvaluatedGenome],
        unique_specimens_selected: usize,
    ) -> N64 {
        self.stats
            .record_run_stat(pre_selection_fitness, population, unique_specimens_selected)
    }

    fn finish_converged(self, final_population: &[EvaluatedGenome]) -> (RunKey, RunStats) {
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

    fn finish_unconverged(self, final_population: &[EvaluatedGenome]) -> (RunKey, RunStats) {
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

// type Genome<const LEN: usize> = BitArr![for LEN, in u16, Lsb0];

// #[derive(Debug, Clone, Copy)]
// struct EvaluatedGenome<const LEN: usize> {
//     genome: Genome<LEN>,
//     fitness: N64,
// }
//
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// enum GenomeDisambiguation<A, B> {
//     Ten(A),
//     Hundo(B),
// }

// type GenomePack = GenomeDisambiguation<Genome<10>, Genome<100>>;
// type GenomeVec = GenomeDisambiguation<Vec<Genome<10>>, Vec<Genome<100>>>;
// type EvaluatedGenomeVec = GenomeDisambiguation<Vec<EvaluatedGenome<10>>, Vec<EvaluatedGenome<100>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genome {
    Ten(BitArr![for 10, in u16, Lsb0]),
    Hundo(BitArr![for 100, in u16, Lsb0]),
}

impl Genome {
    fn empty_ten() -> Self {
        Genome::Ten(bitarr![u16, Lsb0; 0; 10])
    }
    fn empty_hundo() -> Self {
        Genome::Hundo(bitarr![u16, Lsb0; 0; 100])
    }
}

impl Deref for Genome {
    type Target = BitSlice<u16, Lsb0>;

    fn deref(&self) -> &Self::Target {
        match self {
            Genome::Ten(genome) => &genome[..10],
            Genome::Hundo(genome) => &genome[..100],
        }
    }
}

impl DerefMut for Genome {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Genome::Ten(genome) => &mut genome[..10],
            Genome::Hundo(genome) => &mut genome[..100],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvaluatedGenome {
    genome: Genome,
    fitness: N64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum GenomeLength {
    Ten = 10,
    Hundo = 100,
}

pub fn random_genome(rng: &mut impl rand::Rng, length: GenomeLength) -> Genome {
    let mut genome = match length {
        GenomeLength::Ten => Genome::empty_ten(),
        GenomeLength::Hundo => Genome::empty_hundo(),
    };
    for idx in 0..(length as usize) {
        genome.set(idx, rng.gen());
    }
    genome
}

fn homougeneousness<'a>(population: impl ExactSizeIterator<Item = &'a Genome>) -> R64 {
    let len = population.len();
    let mut bit_sum = [Finite::from(0.0); 100];
    let mut max_bit_len = 0;
    for genome in population {
        for (i, bit) in genome.iter().enumerate() {
            if *bit {
                bit_sum[i] += 1.0;
            }
            max_bit_len = std::cmp::max(max_bit_len, i);
        }
    }
    let bit_sum = &mut bit_sum[..=max_bit_len];
    for bit_count in &mut *bit_sum {
        *bit_count =
            Finite::abs(*bit_count / (Finite::from_usize(len).unwrap()) - Finite::from(0.5))
                + Finite::from(0.5);
    }

    *bit_sum.iter().min().expect("Population not to be empty")
}

#[test]
fn same_genomes_are_homougeneous() {
    use bitvec::{bitarr, order::Lsb0};

    let population = vec![
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
    ];
    assert_eq!(homougeneousness(population.iter()), 1.0);
}

#[test]
fn almost_identical_gemoes_are_somewhat_homougeneous() {
    use bitvec::{bitarr, order::Lsb0};

    let population = vec![
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 1, 0, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 1, 0, 0, 1, 1, 0, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 0, 0, 1]),
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 0, 1, 0, 1, 0, 0]),
    ];
    assert_eq!(homougeneousness(population.iter()), 0.8);
}

const HOMOUGENEOUSNESS_THRESHOLD: f64 = 0.99;

fn avg_fitness(population: &[EvaluatedGenome]) -> N64 {
    population.iter().map(|g| g.fitness).sum::<N64>() / N64::from_usize(population.len()).unwrap()
}

fn simulation(mut state: RunState) -> (RunKey, RunStats) {
    let starting_population = state.initial_population();
    let mut starting_population = evaluate(&mut state, starting_population);
    let mut avg_fitness = avg_fitness(&starting_population);
    for _ in 0..MAX_GENERATIONS {
        let SelectionResult {
            new_population: population,
            unique_specimens_selected,
        } = selection(&mut state, starting_population);
        if is_convergant(&state, &population) {
            return state.finish_converged(&population);
        }
        avg_fitness = state.record_run_stat(avg_fitness, &population, unique_specimens_selected);
        let population = crossover(&mut state, population);
        let population = mutation(&mut state, population);
        let population = evaluate(&mut state, population);
        starting_population = population;
    }
    return state.finish_unconverged(&starting_population);
}

fn is_convergant(state: &RunState, population: &[EvaluatedGenome]) -> bool {
    if state.config.mutation_rate.is_none() {
        return is_all_same(population.iter().map(|g| &g.genome));
    }
    return homougeneousness(population.iter().map(|g| &g.genome)) > HOMOUGENEOUSNESS_THRESHOLD;
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

fn config_permutations() -> Vec<AlgoConfig> {
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
                ty: AlgoType::HasPhoenotype(algo, encoding),
                population_size,
                apply_crossover,
                mutation_rate: if apply_mutation { Some(mutation_rate) } else { None },
                selection: Selection::StochasticTournament {
                    prob: selection_prob.into(),
                    replacement,
                },
                genome_len: pow1::GENE_LENGTH,
                optimal_specimen: Some(optimal_specimen),
            });
        }
    }

    perms! {
        (population_size, mutation_rate) in [(100, 0.00001)];
        apply_crossover in [true, false];
        apply_mutation in [true, false];
        selection_prob in [1.0, 0.8, 0.7, 0.6];
        replacement in [TournamentReplacement::With, TournamentReplacement::Without];
        algo in [BinaryAlgo::FConst, BinaryAlgo::FHD { sigma: 100.0.into() }];
        {
            res.push(AlgoConfig {
                ty: AlgoType::BinaryOnly(algo),
                population_size,
                apply_crossover,
                mutation_rate: if apply_mutation { Some(mutation_rate) } else { None },
                selection: Selection::StochasticTournament {
                    prob: selection_prob.into(),
                    replacement,
                },
                genome_len: GenomeLength::Hundo,
                optimal_specimen: optimal_binary_specimen(algo),
            });
        }
    };

    res
}

fn main() {
    let configs = config_permutations();
    let len = configs.len() * 100;
    println!("Running {} configs", configs.len());
    let counter = AtomicUsize::new(0);

    #[cfg(feature = "parallel")]
    rayon::scope(|s| {
        for (i, config) in configs.into_iter().enumerate() {
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
                for run_idx in 0..MAX_RUNS {
                    let state = RunState::new(&config, run_idx);
                    let (_key, _stats) = simulation(state);
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            });
        }
    });

    #[cfg(not(feature = "parallel"))]
    for (i, config) in configs.into_iter().enumerate() {
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
        for run_idx in 0..MAX_RUNS {
            let state = RunState::new(&config, run_idx);
            let (_key, _stats) = simulation(state);
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}
