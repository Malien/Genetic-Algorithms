#![feature(iter_array_chunks)]
#![feature(generic_const_exprs)]
#![feature(type_changing_struct_update)]
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
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
use reports::Report;
use selection::{selection, Selection, SelectionResult, TournamentReplacement};
use stats::{ConfigStats, RunStats, StatEncoder, SuccessFamily};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

use crate::{
    graphs::draw_graphs,
    persistance::{append_config, create_config_writer, write_graphs},
    reports::run_reports,
};

const MAX_GENERATIONS: usize = 10_000_000;
const MAX_RUNS: usize = 100;

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

    fn record_population(&mut self, population: &[EvaluatedGenome<{ F::N }>]) {
        self.stats.record_population(population)
    }

    fn finish_converged(self, final_population: &[EvaluatedGenome<{ F::N }>]) -> RunStats {
        self.stats.finish_converged(final_population)
    }

    fn finish_unconverged(self, final_population: &[EvaluatedGenome<{ F::N }>]) -> RunStats {
        self.stats.finish_unconverged(final_population)
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
    state.record_population(&starting_population);
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
        state.record_population(&population);
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
        (population_size, mutation_rate) in [
            (100, 0.00001),
            // (200, 0.00001 / 2.0),
            // (300, 0.00001 / 3.0),
            // (400, 0.00001 / 4.0),
            // (500, 0.00001 / 5.0),
            // (1000, 0.00001 / 10.0),
        ];
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
        (population_size, mutation_rate) in [
            (100, 0.0005),
            // (200, 0.0005 / 2.0),
            // (300, 0.0005 / 3.0),
            // (400, 0.0005 / 4.0),
            // (500, 0.0005 / 5.0),
            // (1000, 0.0005 / 10.0)
        ];
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
    config_writer: Arc<tokio::sync::Mutex<CSVFile>>,
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

    let _ = state.report_chan.send(Report::solving(thread_id, key));
    let runs = (0..MAX_RUNS)
        .map(|run_idx| {
            let run_state = RunState::new(&config, run_idx);
            simulation(run_state)
        })
        .collect();
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
                let _ = report_chan.send(Report::wrote_tables(key));
            }
        },
        &state.runtime,
    );

    write_tasks.spawn_on(
        {
            let stats = Arc::clone(&stats);
            let config_writer = Arc::clone(&state.config_writer);
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
        let _ = state.report_chan.send(Report::graphing(thread_id, key));
        let graphs = draw_graphs(&config.ty, stats.as_ref()).expect("Drawing to succeed");
        let _ = state.report_chan.send(Report::end(thread_id, key));

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
                    let _ = report_chan.send(Report::wrote_graphs(key));
                }
            },
            &state.runtime,
        );
    }
}

fn main() {
    color_eyre::install().unwrap();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_keep_alive(Duration::from_secs(100))
        .build()
        .unwrap();
    let config_10 = config_permutations_10();
    let config_100 = config_permutations_100();
    // let config_100: Vec<AlgoConfig<G100>> = vec![];
    // let config_10 = vec![AlgoConfig::<G10> {
    //     ty: (PheonotypeAlgo::Pow2, GenomeEncoding::Binary),
    //     population_size: 100,
    //     apply_crossover: true,
    //     mutation_rate: Some(0.0005),
    //     selection: Selection::StochasticTournament {
    //         prob: 1.0.into(),
    //         replacement: TournamentReplacement::With,
    //     },
    //     optimal_specimen: Some(evaluation::pow2::optimal_specimen()),
    // }];
    let config_writer = runtime.block_on(create_config_writer()).unwrap();
    let (tx, rx) = unbounded_channel();
    let program_state = ProgramState {
        file_limiter: Arc::new(tokio::sync::Semaphore::new(10)),
        config_writer: Arc::new(tokio::sync::Mutex::new(config_writer)),
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
            let mut config_writer = wait_for_arc(program_state.config_writer).await.into_inner();
            config_writer.flush().await.unwrap();
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
