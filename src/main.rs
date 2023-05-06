#![feature(iter_array_chunks)]
#![feature(generic_const_exprs)]
#![feature(type_changing_struct_update)]
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use bitvec::{order::Lsb0, slice::BitSlice, BitArr};
use decorum::{Finite, N64, R64};
use eyre::WrapErr;
use graphs::GraphDescriptor;
use num_traits::{real::Real, FromPrimitive};
use rand::rngs::StdRng;

mod evaluation;
mod graphs;
mod operators;
mod persistance;
mod reports;
mod selection;
mod stats;

use evaluation::{
    binary_to_gray, evaluate, optimal_binary_specimen, optimal_pheonotype_specimen, BinaryAlgo,
    EvaluateFamily, GenomeEncoding, PheonotypeAlgo,
};
use operators::{crossover, mutation};
use persistance::{write_stats, CSVFile, ConfigKey};
use reports::{report, Report};
use selection::{selection, Selection, TournamentReplacement};
use stats::{ConfigStats, RunStats, SelectionStats, StatEncoder, SuccessFamily};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

use crate::{
    graphs::draw_graphs,
    persistance::{append_config, create_multi_config_writer, write_graphs},
    reports::run_reports,
};

// const MAX_GENERATIONS: usize = 10_000_000;
const MAX_GENERATIONS: usize = 10_000;
const MAX_RUNS: usize = 100;
const STALL_THRESHOLD: usize = 2_000;
const MAX_FAILED_RUNS: usize = 10;

#[derive(Debug, Clone, Copy)]
pub struct AlgoConfig<F: GenFamily + ?Sized>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    pub ty: F::AlgoType,
    pub population_size: usize,
    pub apply_crossover: bool,
    pub mutation_rate: Option<f64>,
    pub selection: Selection,
    pub optimal_specimen: Option<Genome<{ F::N }>>,
}

impl<F: GenFamily + ?Sized> AlgoConfig<F>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    fn to_key(&self) -> ConfigKey<F::AlgoType> {
        ConfigKey {
            algo_type: self.ty,
            population_size: self.population_size,
            apply_crossover: self.apply_crossover,
            apply_mutation: self.mutation_rate.is_some(),
            selection: self.selection,
        }
    }
}

pub struct RunState<'a, F: GenFamily + ?Sized>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    config: &'a AlgoConfig<F>,
    rng: StdRng,
    report_chan: &'a UnboundedSender<Report>,
}

impl<'a, F: FullFamily> RunState<'a, F>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    fn new(
        config: &'a AlgoConfig<F>,
        report_chan: &'a UnboundedSender<Report>,
        run_idx: usize,
    ) -> Self {
        Self {
            config,
            rng: rand::SeedableRng::seed_from_u64(run_idx as u64),
            report_chan,
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

    fn report_stall(&self, progress: usize, out_of: usize) {
        report(
            Report::taking_too_long(
                std::thread::current().id(),
                self.config.to_key(),
                (progress, out_of),
            ),
            &self.report_chan,
        )
    }
}

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

pub trait AlgoDescriptor {
    fn name(&self) -> &'static str;
    fn category(&self) -> &'static str;
    fn materialize(&self) -> MaterializedDescriptor {
        MaterializedDescriptor {
            category: self.category(),
            name: self.name(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaterializedDescriptor {
    category: &'static str,
    name: &'static str,
}

impl AlgoDescriptor for MaterializedDescriptor {
    fn category(&self) -> &'static str {
        self.category
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

impl AlgoDescriptor for BinaryAlgo {
    fn name(&self) -> &'static str {
        match self {
            BinaryAlgo::FConst => "FConst",
            BinaryAlgo::FHD { .. } => "FHD",
        }
    }
    fn category(&self) -> &'static str {
        "binary"
    }
}

impl AlgoDescriptor for (PheonotypeAlgo, GenomeEncoding) {
    fn name(&self) -> &'static str {
        match self.0 {
            PheonotypeAlgo::Pow1 => "x^2",
            PheonotypeAlgo::Pow2 => "5.12^2 - x^2",
        }
    }
    fn category(&self) -> &'static str {
        match self.1 {
            GenomeEncoding::Binary => "real",
            GenomeEncoding::BinaryGray => "real-gray",
        }
    }
}

pub trait GenFamily {
    const N: usize;
    type AlgoType: Copy + AlgoDescriptor;
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

fn simulation<F: FullFamily>(mut state: RunState<F>) -> RunStats
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    let starting_population = state.initial_population();
    let mut starting_population = evaluate(&mut state, starting_population);
    let mut stats = StatEncoder::new(state.config);
    stats.record_population(&starting_population);
    let mut avg_fitness = avg_fitness(&starting_population);
    for gen in 1..=MAX_GENERATIONS {
        if is_convergant(&state, &starting_population) {
            return stats.finish_converged(&starting_population);
        }
        if gen % STALL_THRESHOLD == 0 {
            state.report_stall(gen / STALL_THRESHOLD, MAX_GENERATIONS / STALL_THRESHOLD);
        }
        let selection_result = selection(&mut state, starting_population);
        let (population, selection_stats) =
            SelectionStats::from_result(selection_result, avg_fitness);

        let population = crossover(&mut state, population);
        let population = mutation(&mut state, population);
        let population = evaluate(&mut state, population);

        avg_fitness = stats.record_run_stat(&population, selection_stats);
        stats.record_population(&population);
        starting_population = population;
    }
    return stats.finish_unconverged(&starting_population);
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

const POPULATION_SIZES: [usize; 6] = [100, 200, 300, 400, 500, 1000];
const MUTATION_MODIFIERS: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
const G10_BASE_MUTATION_RATE: f64 = 0.0001;
const G100_BASE_MUTATION_RATE: f64 = 0.00001;

fn config_permutations_100() -> Vec<AlgoConfig<G100>> {
    let mut res = vec![];

    perms! {
        (population_size, mutation_rate) in POPULATION_SIZES
            .into_iter()
            .zip(MUTATION_MODIFIERS.into_iter().map(|x| x * G100_BASE_MUTATION_RATE))
            .rev();
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
        (population_size, mutation_rate) in POPULATION_SIZES
            .into_iter()
            .zip(MUTATION_MODIFIERS.into_iter().map(|x| x * G10_BASE_MUTATION_RATE))
            .rev();
        apply_crossover in [true, false];
        apply_mutation in [true, false];
        selection_prob in [1.0, 0.8, 0.7, 0.6];
        replacement in [TournamentReplacement::With, TournamentReplacement::Without];
        algo in [PheonotypeAlgo::Pow1, PheonotypeAlgo::Pow2];
        (encoding, optimal_specimen) in [
            (GenomeEncoding::Binary, optimal_pheonotype_specimen(algo)),
            (GenomeEncoding::BinaryGray, binary_to_gray(optimal_pheonotype_specimen(algo))),
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

struct ProgramState {
    file_limiter: Arc<tokio::sync::Semaphore>,
    runtime: tokio::runtime::Handle,
    config_writers: HashMap<usize, Arc<tokio::sync::Mutex<CSVFile>>>,
    write_tasks: std::sync::Mutex<tokio::task::JoinSet<()>>,
    len: usize,
    report_chan: UnboundedSender<Report>,
}

fn run_config<F: FullFamily>(state: &ProgramState, config: AlgoConfig<F>)
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
    F::AlgoType: GraphDescriptor + Send + Sync + 'static,
{
    let thread_id = std::thread::current().id();
    let key = config.to_key();

    report(Report::solving(thread_id, key), &state.report_chan);
    let mut runs = Vec::with_capacity(MAX_RUNS);
    let mut failed_runs = Some(0);
    for run_idx in 0..MAX_RUNS {
        let run_state = RunState::new(&config, &state.report_chan, run_idx);
        let run = simulation(run_state);
        let converged = run.converged();
        runs.push(run);
        match (&mut failed_runs, converged) {
            (Some(failed_runs), false) if *failed_runs >= MAX_FAILED_RUNS => {
                report(Report::cutting_short(key, *failed_runs), &state.report_chan);
                break;
            }
            (Some(failed_runs), false) => {
                *failed_runs += 1;
            }
            (Some(_), true) => {
                failed_runs = None;
            }
            _ => {}
        }
    }
    let stats = Arc::new(ConfigStats::new(runs));

    let mut write_tasks = state.write_tasks.lock().unwrap();
    write_tasks.spawn_on(
        {
            let stats = Arc::clone(&stats);
            let file_limiter = Arc::clone(&state.file_limiter);
            let report_chan = state.report_chan.clone();
            async move {
                write_stats(file_limiter.as_ref(), key, stats.as_ref())
                    .await
                    .wrap_err_with(|| format!("Writing stats {key}"))
                    .unwrap();
                report(Report::wrote_tables(key), &report_chan);
            }
        },
        &state.runtime,
    );

    write_tasks.spawn_on(
        {
            let stats = Arc::clone(&stats);
            let config_writer = Arc::clone(&state.config_writers[&config.population_size]);
            async move {
                let mut writer = config_writer.lock().await;
                append_config(writer.deref_mut(), key, stats.as_ref())
                    .await
                    .wrap_err_with(|| format!("Appending config {key}"))
                    .unwrap();
            }
        },
        &state.runtime,
    );
    drop(write_tasks);

    #[cfg(not(feature = "drop_graphs"))]
    {
        report(Report::graphing(thread_id, key), &state.report_chan);
        let graphs = draw_graphs(&config.ty, stats.as_ref()).expect("Drawing to succeed");
        report(Report::end(thread_id, key), &state.report_chan);

        let mut write_tasks = state.write_tasks.lock().unwrap();
        write_tasks.spawn_on(
            {
                let file_limiter = Arc::clone(&state.file_limiter);
                let report_chan = state.report_chan.clone();
                async move {
                    write_graphs(file_limiter.as_ref(), key, graphs)
                        .await
                        .wrap_err_with(|| format!("Writing graphs of {key}"))
                        .unwrap();
                    report(Report::wrote_graphs(key), &report_chan);
                }
            },
            &state.runtime,
        );
    }
}

pub static VERBOSE: AtomicBool = AtomicBool::new(false);
pub static SILENT: AtomicBool = AtomicBool::new(false);

fn main() {
    color_eyre::install().unwrap();

    for arg in std::env::args() {
        if arg == "--verbose" {
            VERBOSE.store(true, Ordering::Relaxed);
        }
        if arg == "--silent" {
            SILENT.store(true, Ordering::Relaxed);
        }
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_keep_alive(Duration::from_secs(100))
        .build()
        .unwrap();

    let config_10 = config_permutations_10();
    let config_100 = config_permutations_100();

    // let config_10: Vec<AlgoConfig<G10>> = vec![];
    // let config_100 = vec![AlgoConfig::<G100> {
    //     ty: BinaryAlgo::FConst,
    //     population_size: 100,
    //     apply_crossover: true,
    //     // mutation_rate: None,
    //     mutation_rate: Some(0.0005),
    //     selection: Selection::StochasticTournament {
    //         prob: 1.0.into(),
    //         replacement: TournamentReplacement::With,
    //     },
    //     optimal_specimen: None,
    // }];

    // let config_100: Vec<AlgoConfig<G100>> = vec![];
    // let config_10 = vec![AlgoConfig::<G10> {
    //     ty: (PheonotypeAlgo::Pow1, GenomeEncoding::Binary),
    //     population_size: 100,
    //     apply_crossover: false,
    //     mutation_rate: Some(0.0001),
    //     selection: Selection::StochasticTournament {
    //         prob: 1.0.into(),
    //         replacement: TournamentReplacement::With,
    //     },
    //     optimal_specimen: Some(evaluation::pow1::optimal_specimen()),
    // }];

    let config_writers = runtime
        .block_on(create_multi_config_writer(POPULATION_SIZES))
        .unwrap();
    let (tx, rx) = unbounded_channel();
    let program_state = ProgramState {
        file_limiter: Arc::new(tokio::sync::Semaphore::new(10)),
        config_writers,
        write_tasks: std::sync::Mutex::new(tokio::task::JoinSet::new()),
        len: config_10.len() + config_100.len(),
        runtime: runtime.handle().clone(),
        report_chan: tx,
    };
    println!("Running {} configs", program_state.len);

    let reporter_task = runtime.spawn(run_reports(program_state.len as u64, rx));

    rayon::scope(|s| {
        for config in config_10 {
            let program_state = &program_state;
            s.spawn(move |_| run_config(program_state, config));
        }
        for config in config_100 {
            let program_state = &program_state;
            s.spawn(move |_| run_config(program_state, config));
        }
    });

    // for config in config_10 {
    //     run_config(&program_state, config);
    // }
    // for config in config_100 {
    //     run_config(&program_state, config);
    // }

    runtime.block_on(async move {
        drop(program_state.report_chan);
        reporter_task.await.unwrap().unwrap();

        let b = async move {
            let mut write_tasks = program_state.write_tasks.into_inner().unwrap();
            while let Some(_) = write_tasks.join_next().await {}
        };
        let a = async move {
            for config_writer in program_state.config_writers.into_values() {
                let mut config_writer = wait_for_arc(config_writer).await.into_inner();
                config_writer.flush().await.unwrap();
            }
        };
        futures::future::join(a, b).await;
    });
}

async fn wait_for_arc<T>(mut arc: Arc<T>) -> T {
    loop {
        match Arc::try_unwrap(arc) {
            Ok(value) => return value,
            Err(same_arc) => {
                arc = same_arc;
                tokio::task::yield_now().await;
            }
        };
    }
}
