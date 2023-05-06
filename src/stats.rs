use std::{
    iter::Sum,
    ops::{Add, Div},
};

use arrayvec::ArrayVec;
use decorum::N64;
use num::FromPrimitive;
use num_traits::real::Real;

use crate::{
    avg_fitness,
    evaluation::{evaluate_phenotype, EvaluateFamily},
    selection::SelectionResult,
    AlgoConfig, EvaluatedGenome, GenFamily, Genome, G10, G100,
};

#[derive(Debug, Clone)]
pub struct IterationStats {
    pub selection_intensity: N64,
    pub growth_rate: N64,
    pub optimal_specimen_count: usize,
    pub avg_fitness: N64,
    pub best_fitness: N64,
    pub fitness_std_dev: N64,
    pub rr: N64,
    pub theta: N64,
    pub selection_diff: N64,
}

#[derive(Debug, Clone, Copy)]
pub struct OptimumlessIterationStats {
    pub avg_fitness: N64,
    pub best_fitness: N64,
    pub fitness_std_dev: N64,
    pub rr: N64,
    pub theta: N64,
    // selection_diff: N64,
}

#[derive(Debug, Clone)]
pub struct PopulationStats {
    pub fitness: Vec<N64>,
    pub ones_count: Vec<usize>,
    pub phenotype: Option<Vec<N64>>,
}

#[derive(Debug, Clone)]
pub struct RunStatsWithOptimum {
    pub iterations: Vec<IterationStats>,
    pub starting_population: ArrayVec<PopulationStats, 5>,
    pub final_population: PopulationStats,

    // pub populations: Vec<Vec<PopulationStats>>,
    pub best_fitness: N64,
    pub avg_fitness: N64,
    pub success: bool,
    pub converged: bool,
    // iteration_count: usize, // is the same as iterations.len()
    pub min_selection_intensity: (usize, N64),
    pub max_selection_intensity: (usize, N64),
    pub avg_selection_intensity: N64,
    pub early_growth_rate: N64,
    pub avg_growth_rate: N64,
    pub late_growth_rate: Option<(usize, N64)>,

    pub min_rr: (usize, N64),
    pub max_rr: (usize, N64),
    pub avg_rr: N64,
    pub min_theta: (usize, N64),
    pub max_theta: (usize, N64),
    pub avg_theta: N64,

    pub min_selection_diff: (usize, N64),
    pub max_selection_diff: (usize, N64),
    pub avg_selection_diff: N64,
}

#[derive(Debug, Clone)]
pub struct OptimumlessRunStats {
    pub iterations: Vec<OptimumlessIterationStats>,
    pub starting_population: ArrayVec<PopulationStats, 5>,
    pub final_population: PopulationStats,

    // best_fitness: N64,
    // avg_fitness: N64,
    pub success: bool,
    pub converged: bool,
    // iteration_count: usize, // is the same as iterations.len()
    pub min_rr: (usize, N64),
    pub max_rr: (usize, N64),
    pub avg_rr: N64,
    pub min_theta: (usize, N64),
    pub max_theta: (usize, N64),
    pub avg_theta: N64,
}

#[derive(Debug, Clone)]
pub struct OptimumlessConfigStats {
    pub runs: Vec<OptimumlessRunStats>,

    pub success_percent: f64,
    // For successful runs only
    pub min_iteration_count: Option<usize>,
    pub max_iteration_count: Option<usize>,
    pub avg_iteration_count: Option<f64>,
    pub iteration_count_std_dev: Option<f64>,

    // For successful runs only
    pub min_min_rr: Option<(usize, N64)>,
    pub max_max_rr: Option<(usize, N64)>,
    pub avg_min_rr: Option<N64>,
    pub avg_max_rr: Option<N64>,
    pub avg_avg_rr: Option<N64>,
    pub min_rr_std_dev: Option<N64>,
    pub max_rr_std_dev: Option<N64>,
    pub avg_rr_std_dev: Option<N64>,

    // For successful runs only
    pub min_min_theta: Option<(usize, N64)>,
    pub max_max_theta: Option<(usize, N64)>,
    pub avg_min_theta: Option<N64>,
    pub avg_max_theta: Option<N64>,
    pub avg_avg_theta: Option<N64>,
    pub min_theta_std_dev: Option<N64>,
    pub max_theta_std_dev: Option<N64>,
    pub avg_theta_std_dev: Option<N64>,
    // // For successful runs only
    // min_min_selection_diff: Option<(usize, N64)>,
    // max_max_selection_diff: Option<(usize, N64)>,
    // avg_min_selection_diff: Option<N64>,
    // avg_max_selection_diff: Option<N64>,
    // avg_avg_selection_diff: Option<N64>,
}

impl OptimumlessConfigStats {
    pub fn new(runs: Vec<OptimumlessRunStats>) -> Self {
        Self {
            success_percent: runs.iter().filter(|r| r.success).count() as f64 / runs.len() as f64
                * 100.0,

            min_iteration_count: runs
                .iter()
                .filter(|r| r.success)
                .map(|r| r.iterations.len())
                .min(),
            max_iteration_count: runs
                .iter()
                .filter(|r| r.success)
                .map(|r| r.iterations.len())
                .max(),
            avg_iteration_count: Self::run_avg(&runs, |r| r.iterations.len() as f64),
            iteration_count_std_dev: Self::run_std_dev(&runs, |r| r.iterations.len() as f64),

            min_min_rr: Self::run_min(&runs, |r| r.min_rr.1),
            max_max_rr: Self::run_max(&runs, |r| r.max_rr.1),
            avg_min_rr: Self::run_avg(&runs, |r| r.min_rr.1),
            avg_max_rr: Self::run_avg(&runs, |r| r.max_rr.1),
            avg_avg_rr: Self::run_avg(&runs, |r| r.avg_rr),
            min_rr_std_dev: Self::run_std_dev(&runs, |r| r.min_rr.1),
            max_rr_std_dev: Self::run_std_dev(&runs, |r| r.max_rr.1),
            avg_rr_std_dev: Self::run_std_dev(&runs, |r| r.avg_rr),

            min_min_theta: Self::run_min(&runs, |r| r.min_theta.1),
            max_max_theta: Self::run_max(&runs, |r| r.max_theta.1),
            avg_min_theta: Self::run_avg(&runs, |r| r.min_theta.1),
            avg_max_theta: Self::run_avg(&runs, |r| r.max_theta.1),
            avg_avg_theta: Self::run_avg(&runs, |r| r.avg_theta),
            min_theta_std_dev: Self::run_std_dev(&runs, |r| r.min_theta.1),
            max_theta_std_dev: Self::run_std_dev(&runs, |r| r.max_theta.1),
            avg_theta_std_dev: Self::run_std_dev(&runs, |r| r.avg_theta),

            // min_min_selection_diff: Self::run_min(&runs, |r| r.min_selection_diff.1),
            // max_max_selection_diff: Self::run_max(&runs, |r| r.max_selection_diff.1),
            // avg_min_selection_diff: Self::run_avg(&runs, |r| r.min_selection_diff.1),
            // avg_max_selection_diff: Self::run_avg(&runs, |r| r.max_selection_diff.1),
            // avg_avg_selection_diff: Self::run_avg(&runs, |r| r.avg_selection_diff),
            runs,
        }
    }

    fn run_min<T: Ord + Copy>(
        runs: &[OptimumlessRunStats],
        selector: impl Fn(&OptimumlessRunStats) -> T,
    ) -> Option<(usize, T)> {
        runs.iter()
            .filter(|r| r.success)
            .map(selector)
            .enumerate()
            .max_by_key(|(_, v)| *v)
    }

    fn run_max<T: Ord + Copy>(
        runs: &[OptimumlessRunStats],
        selector: impl Fn(&OptimumlessRunStats) -> T,
    ) -> Option<(usize, T)> {
        runs.iter()
            .filter(|r| r.success)
            .map(selector)
            .enumerate()
            .max_by_key(|(_, v)| *v)
    }

    fn run_avg<T>(
        runs: &[OptimumlessRunStats],
        selector: impl Fn(&OptimumlessRunStats) -> T,
    ) -> Option<T>
    where
        T: Copy + Add<Output = T> + Div<f64, Output = T>,
    {
        runs.iter()
            .filter(|r| r.success)
            .map(selector)
            .fold(None, |acc, x| match acc {
                None => Some(x.clone()),
                Some(y) => Some(x + y),
            })
            .map(|x| x / runs.len() as f64)
    }

    fn run_std_dev<T>(
        runs: &[OptimumlessRunStats],
        selector: impl Fn(&OptimumlessRunStats) -> T,
    ) -> Option<T>
    where
        T: Copy + Div<f64, Output = T> + Real,
    {
        let avg = Self::run_avg(runs, &selector)?;
        let sum = runs
            .iter()
            .filter(|r| r.success)
            .map(|r| {
                let x = selector(r);
                (x - avg) * (x - avg)
            })
            .fold(None, |acc, x| match acc {
                None => Some(x.clone()),
                Some(y) => Some(x + y),
            })?;
        Some(sum.sqrt())
    }
}

#[derive(Debug, Clone)]
pub struct ConfigStatsWithOptimum {
    pub runs: Vec<RunStatsWithOptimum>,

    pub success_percent: f64,
    // For successful runs only
    pub min_iteration_count: Option<usize>,
    pub max_iteration_count: Option<usize>,
    pub avg_iteration_count: Option<f64>,
    pub iteration_count_std_dev: Option<f64>,

    pub min_min_selection_intensity: (usize, N64),
    pub max_max_selection_intensity: (usize, N64),
    pub avg_min_selection_intensity: N64,
    pub avg_max_selection_intensity: N64,
    pub avg_selection_intensity: N64,
    pub min_selection_intensity_std_dev: N64,
    pub max_selection_intensity_std_dev: N64,
    pub avg_selection_intensity_std_dev: N64,

    pub min_early_growth_rate: N64,
    pub max_early_growth_rate: N64,
    pub avg_early_growth_rate: N64,
    pub min_late_growth_rate: Option<N64>,
    pub max_late_growth_rate: Option<N64>,
    pub avg_late_growth_rate: Option<N64>,
    pub min_avg_growth_rate: N64,
    pub max_avg_growth_rate: N64,
    pub avg_avg_growth_rate: N64,

    // For successful runs only
    pub min_min_rr: Option<(usize, N64)>,
    pub max_max_rr: Option<(usize, N64)>,
    pub avg_min_rr: Option<N64>,
    pub avg_max_rr: Option<N64>,
    pub avg_avg_rr: Option<N64>,
    pub min_rr_std_dev: Option<N64>,
    pub max_rr_std_dev: Option<N64>,
    pub avg_rr_std_dev: Option<N64>,

    // For successful runs only
    pub min_min_theta: Option<(usize, N64)>,
    pub max_max_theta: Option<(usize, N64)>,
    pub avg_min_theta: Option<N64>,
    pub avg_max_theta: Option<N64>,
    pub avg_avg_theta: Option<N64>,
    pub min_theta_std_dev: Option<N64>,
    pub max_theta_std_dev: Option<N64>,
    pub avg_theta_std_dev: Option<N64>,
}

fn optinal_avg<T>(values: &[T]) -> Option<T>
where
    T: Copy + Add<Output = T> + Div<f64, Output = T>,
{
    values
        .iter()
        .fold(None, |acc, x| match acc {
            None => Some(x.clone()),
            Some(y) => Some(*x + y),
        })
        .map(|x| x / values.len() as f64)
}

impl ConfigStatsWithOptimum {
    pub fn new(runs: Vec<RunStatsWithOptimum>) -> Self {
        let late_growth_rates: Vec<_> = runs
            .iter()
            .filter_map(|r| r.late_growth_rate)
            .map(|(_, r)| r)
            .collect();

        Self {
            success_percent: runs.iter().filter(|r| r.success).count() as f64 / runs.len() as f64
                * 100.0,
            min_iteration_count: runs
                .iter()
                .filter(|r| r.success)
                .map(|r| r.iterations.len())
                .min(),
            max_iteration_count: runs
                .iter()
                .filter(|r| r.success)
                .map(|r| r.iterations.len())
                .max(),
            avg_iteration_count: Self::run_avg(&runs, |r| r.iterations.len() as f64),
            iteration_count_std_dev: Self::run_std_dev(&runs, |r| r.iterations.len() as f64),

            min_min_selection_intensity: iteration_min(&runs, |r| r.min_selection_intensity.1),
            max_max_selection_intensity: iteration_max(&runs, |r| r.max_selection_intensity.1),
            avg_min_selection_intensity: avg_by(&runs, |r| r.min_selection_intensity.1),
            avg_max_selection_intensity: avg_by(&runs, |r| r.max_selection_intensity.1),
            avg_selection_intensity: avg_by(&runs, |r| r.avg_selection_intensity),
            min_selection_intensity_std_dev: std_dev_by(&runs, |r| r.min_selection_intensity.1),
            max_selection_intensity_std_dev: std_dev_by(&runs, |r| r.max_selection_intensity.1),
            avg_selection_intensity_std_dev: std_dev_by(&runs, |r| r.avg_selection_intensity),

            min_early_growth_rate: runs
                .iter()
                .map(|r| r.early_growth_rate)
                .min()
                .expect("At least one run recorded"),
            max_early_growth_rate: runs
                .iter()
                .map(|r| r.early_growth_rate)
                .max()
                .expect("At least one run recorded"),
            avg_early_growth_rate: avg_by(&runs, |r| r.early_growth_rate),
            min_late_growth_rate: late_growth_rates.iter().min().copied(),
            max_late_growth_rate: late_growth_rates.iter().min().copied(),
            avg_late_growth_rate: optinal_avg(&late_growth_rates),
            min_avg_growth_rate: runs
                .iter()
                .map(|r| r.avg_growth_rate)
                .min()
                .expect("At least one run recorded"),
            max_avg_growth_rate: runs
                .iter()
                .map(|r| r.avg_growth_rate)
                .max()
                .expect("At least one run recorded"),
            avg_avg_growth_rate: avg_by(&runs, |r| r.avg_growth_rate),

            min_min_rr: Self::run_min(&runs, |r| r.min_rr.1),
            max_max_rr: Self::run_max(&runs, |r| r.max_rr.1),
            avg_min_rr: Self::run_avg(&runs, |r| r.min_rr.1),
            avg_max_rr: Self::run_avg(&runs, |r| r.max_rr.1),
            avg_avg_rr: Self::run_avg(&runs, |r| r.avg_rr),
            min_rr_std_dev: Self::run_std_dev(&runs, |r| r.min_rr.1),
            max_rr_std_dev: Self::run_std_dev(&runs, |r| r.max_rr.1),
            avg_rr_std_dev: Self::run_std_dev(&runs, |r| r.avg_rr),

            min_min_theta: Self::run_min(&runs, |r| r.min_theta.1),
            max_max_theta: Self::run_max(&runs, |r| r.max_theta.1),
            avg_min_theta: Self::run_avg(&runs, |r| r.min_theta.1),
            avg_max_theta: Self::run_avg(&runs, |r| r.max_theta.1),
            avg_avg_theta: Self::run_avg(&runs, |r| r.avg_theta),
            min_theta_std_dev: Self::run_std_dev(&runs, |r| r.min_theta.1),
            max_theta_std_dev: Self::run_std_dev(&runs, |r| r.max_theta.1),
            avg_theta_std_dev: Self::run_std_dev(&runs, |r| r.avg_theta),

            runs,
        }
    }

    fn run_min<T: Ord + Copy>(
        runs: &[RunStatsWithOptimum],
        selector: impl Fn(&RunStatsWithOptimum) -> T,
    ) -> Option<(usize, T)> {
        runs.iter()
            .filter(|r| r.success)
            .map(selector)
            .enumerate()
            .max_by_key(|(_, v)| *v)
    }

    fn run_max<T: Ord + Copy>(
        runs: &[RunStatsWithOptimum],
        selector: impl Fn(&RunStatsWithOptimum) -> T,
    ) -> Option<(usize, T)> {
        runs.iter()
            .filter(|r| r.success)
            .map(selector)
            .enumerate()
            .max_by_key(|(_, v)| *v)
    }

    fn run_avg<T>(
        runs: &[RunStatsWithOptimum],
        selector: impl Fn(&RunStatsWithOptimum) -> T,
    ) -> Option<T>
    where
        T: Copy + Add<Output = T> + Div<f64, Output = T>,
    {
        runs.iter()
            .filter(|r| r.success)
            .map(selector)
            .fold(None, |acc, x| match acc {
                None => Some(x.clone()),
                Some(y) => Some(x + y),
            })
            .map(|x| x / runs.len() as f64)
    }

    fn run_std_dev<T>(
        runs: &[RunStatsWithOptimum],
        selector: impl Fn(&RunStatsWithOptimum) -> T,
    ) -> Option<T>
    where
        T: Copy + Div<f64, Output = T> + Real,
    {
        let avg = Self::run_avg(runs, &selector)?;
        let sum = runs
            .iter()
            .filter(|r| r.success)
            .map(|r| {
                let x = selector(r);
                (x - avg) * (x - avg)
            })
            .fold(None, |acc, x| match acc {
                None => Some(x.clone()),
                Some(y) => Some(x + y),
            })?;
        Some(sum.sqrt())
    }
}

fn std_dev_by<U, T>(xs: &[U], selector: impl Fn(&U) -> T) -> T
where
    T: Copy + Sum + Div<f64, Output = T> + Real,
{
    let avg = avg_by(xs, &selector);
    let sum = xs
        .iter()
        .map(|x| {
            let x = selector(x);
            (x - avg) * (x - avg)
        })
        .sum::<T>();
    sum.sqrt()
}

pub type ConfigStats = OptimumDisabiguity<ConfigStatsWithOptimum, OptimumlessConfigStats>;

impl ConfigStats {
    pub fn new(runs: Vec<RunStats>) -> Self {
        use OptimumDisabiguity::*;
        match runs.first() {
            Some(WithOptimum(_)) => {
                let runs = runs
                    .into_iter()
                    .map(|r| match r {
                        WithOptimum(r) => r,
                        Optimumless(_) => panic!("Expected all runs to be WithOptimum"),
                    })
                    .collect::<Vec<_>>();
                WithOptimum(ConfigStatsWithOptimum::new(runs))
            }
            Some(Optimumless(_)) => {
                let runs = runs
                    .into_iter()
                    .map(|r| match r {
                        WithOptimum(_) => panic!("Expected all runs to be Optimumless"),
                        Optimumless(r) => r,
                    })
                    .collect::<Vec<_>>();
                Optimumless(OptimumlessConfigStats::new(runs))
            }
            None => panic!("No runs to summarize"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OptimumDisabiguity<With, Without> {
    WithOptimum(With),
    Optimumless(Without),
}

pub type RunStats = OptimumDisabiguity<RunStatsWithOptimum, OptimumlessRunStats>;

impl RunStats {
    pub fn converged(&self) -> bool {
        use OptimumDisabiguity::*;
        match self {
            WithOptimum(run) => run.converged,
            Optimumless(run) => run.converged,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SelectionStats {
    rr: N64,
    theta: N64,
    selection_diff: N64,
    selection_intensity: N64,
}

impl SelectionStats {
    pub fn from_result<const N: usize>(
        selection_result: SelectionResult<N>,
        pre_selection_fitness: N64,
    ) -> (Vec<EvaluatedGenome<N>>, SelectionStats)
    where
        [(); bitvec::mem::elts::<u16>(N)]:,
    {
        let population = selection_result.new_population;
        let unique_specimens_selected =
            N64::from_usize(selection_result.unique_specimens_selected).unwrap();
        let population_size = N64::from_usize(population.len()).unwrap();

        let avg_fitness = avg_fitness(&population);
        let std_dev = std_dev_by(&population, |g| g.fitness);

        let selection_intensity = if std_dev == N64::from(0.0) {
            1.0.into()
        } else {
            (avg_fitness - pre_selection_fitness) / std_dev
        };
        let rr = unique_specimens_selected / population_size;
        let theta = (population_size - unique_specimens_selected) / population_size;
        let selection_diff = avg_fitness - pre_selection_fitness;

        (
            population,
            SelectionStats {
                selection_intensity,
                selection_diff,
                rr,
                theta,
            },
        )
    }
}

pub type StatEncoder<'a, F> =
    OptimumDisabiguity<StateEncoderWithOptimum<'a, F>, OptimumlessStateEncoder>;

pub struct StateEncoderWithOptimum<'a, F: GenFamily + ?Sized>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    config: &'a AlgoConfig<F>,
    optimal_specimen: Genome<{ F::N }>,
    iterations: Vec<IterationStats>,
    prev_iteration_optimal_specimens: usize,
    populations: ArrayVec<PopulationStats, 5>,
}

impl<F: SuccessFamily + EvaluateFamily> StateEncoderWithOptimum<'_, F>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    pub fn record_run_stat(
        &mut self,
        population: &[EvaluatedGenome<{ F::N }>],
        selection_stats: SelectionStats,
    ) -> N64 {
        let optimal_specimen_count = population
            .iter()
            .filter(|genome| genome.genome == self.optimal_specimen)
            .count();
        let population_size = N64::from_usize(population.len()).unwrap();
        let avg_fitness = avg_fitness(population);
        let std_dev = population
            .iter()
            .map(|g| N64::powi(g.fitness - avg_fitness, 2))
            .sum::<N64>()
            / population_size;

        let growth_rate =
            if self.prev_iteration_optimal_specimens != 0 && optimal_specimen_count != 0 {
                N64::from_usize(optimal_specimen_count).unwrap()
                    / N64::from_usize(self.prev_iteration_optimal_specimens).unwrap()
            } else {
                0.0.into()
            };

        self.iterations.push(IterationStats {
            selection_intensity: selection_stats.selection_intensity,
            selection_diff: selection_stats.selection_diff,
            growth_rate,
            optimal_specimen_count,
            avg_fitness,
            best_fitness: population
                .iter()
                .map(|g| g.fitness)
                .max()
                .expect("At least one iteration to be recorded"),
            fitness_std_dev: std_dev,
            rr: selection_stats.rr,
            theta: selection_stats.theta,
        });
        self.prev_iteration_optimal_specimens = optimal_specimen_count;
        return avg_fitness;
    }

    pub fn record_population(&mut self, population: &[EvaluatedGenome<{ F::N }>]) {
        if self.populations.is_full() {
            return;
        }
        self.populations.push(PopulationStats {
            fitness: population.iter().map(|g| g.fitness).collect(),
            ones_count: population.iter().map(|g| g.genome.count_ones()).collect(),
            phenotype: if F::has_phenotype() {
                Some(
                    population
                        .iter()
                        .map(|g| F::decode_phenotype(g.genome, self.config.ty)
                        .expect("if EvaluateFamily::has_phenotype is true, every possible genome has associated phenotype"))
                        .collect(),
                )
            } else { None },
        });
    }

    pub fn finish_converged(
        self,
        final_population: &[EvaluatedGenome<{ F::N }>],
    ) -> RunStatsWithOptimum {
        let success =
            F::is_success_converged(&self.config, final_population, self.optimal_specimen);
        self.finish(final_population, success, true)
    }

    pub fn finish_unconverged(
        self,
        final_population: &[EvaluatedGenome<{ F::N }>],
    ) -> RunStatsWithOptimum {
        let success =
            F::is_success_unconverged(self.config, final_population, self.optimal_specimen);
        self.finish(final_population, success, false)
    }

    fn finish(
        self,
        final_population: &[EvaluatedGenome<{ F::N }>],
        success: bool,
        converged: bool,
    ) -> RunStatsWithOptimum {
        let final_population_stats = PopulationStats {
            fitness: final_population.iter().map(|g| g.fitness).collect(),
            ones_count: final_population
                .iter()
                .map(|g| g.genome.count_ones())
                .collect(),
            phenotype: if F::has_phenotype() {
                Some(
                    final_population
                        .iter()
                        .map(|g| F::decode_phenotype(g.genome, self.config.ty)
                        .expect("if EvaluateFamily::has_phenotype is true, every possible genome has associated phenotype"))
                        .collect(),
                )
            } else {
                None
            },
        };

        RunStatsWithOptimum {
            best_fitness: final_population
                .iter()
                .map(|g| g.fitness)
                .max()
                .expect("Population is not empty"),
            avg_fitness: avg_fitness(&final_population),
            converged,
            success,
            min_selection_intensity: iteration_min(&self.iterations, |i| i.selection_intensity),
            max_selection_intensity: iteration_max(&self.iterations, |i| i.selection_intensity),
            avg_selection_intensity: avg_by(&self.iterations, |i| i.selection_intensity),
            early_growth_rate: self
                .iterations
                .get(1)
                .expect("At least one iteration to be recorded")
                .growth_rate,
            avg_growth_rate: avg_by(&self.iterations, |i| i.growth_rate),
            late_growth_rate: self
                .iterations
                .iter()
                .enumerate()
                .find(|(_idx, iteration)| {
                    iteration.optimal_specimen_count >= self.config.population_size / 2
                })
                .map(|(idx, iteration)| (idx, iteration.growth_rate)),
            min_rr: iteration_min(&self.iterations, |i| i.rr),
            max_rr: iteration_max(&self.iterations, |i| i.rr),
            avg_rr: avg_by(&self.iterations, |i| i.rr),
            min_theta: iteration_min(&self.iterations, |i| i.theta),
            max_theta: iteration_max(&self.iterations, |i| i.theta),
            avg_theta: avg_by(&self.iterations, |i| i.theta),
            min_selection_diff: iteration_min(&self.iterations, |i| i.selection_diff),
            max_selection_diff: iteration_max(&self.iterations, |i| i.selection_diff),
            avg_selection_diff: avg_by(&self.iterations, |i| i.selection_diff),

            iterations: self.iterations,
            starting_population: self.populations,
            final_population: final_population_stats,
        }
    }

}

fn iteration_min<U, T: Ord + Copy>(items: &[U], selector: impl Fn(&U) -> T) -> (usize, T) {
    items
        .iter()
        .map(selector)
        .enumerate()
        .min_by_key(|(_idx, value)| *value)
        .expect("At least one iteration to be recorded")
}

fn iteration_max<U, T: Ord + Copy>(items: &[U], selector: impl Fn(&U) -> T) -> (usize, T) {
    items
        .iter()
        .map(selector)
        .enumerate()
        .max_by_key(|(_idx, value)| *value)
        .expect("At least one iteration to be recorded")
}

fn avg_by<U, T>(items: &[U], selector: impl Fn(&U) -> T) -> T
where
    T: Sum + Div<f64, Output = T>,
{
    items.iter().map(selector).sum::<T>() / items.len() as f64
}

pub trait SuccessFamily: GenFamily {
    fn is_success_converged(
        config: &AlgoConfig<Self>,
        final_population: &[EvaluatedGenome<{ Self::N }>],
        optimal_specimen: Genome<{ Self::N }>,
    ) -> bool
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:;

    fn is_success_unconverged(
        config: &AlgoConfig<Self>,
        final_population: &[EvaluatedGenome<{ Self::N }>],
        optimal_specimen: Genome<{ Self::N }>,
    ) -> bool
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:;
}

impl SuccessFamily for G10 {
    fn is_success_converged(
        config: &AlgoConfig<Self>,
        final_population: &[EvaluatedGenome<{ Self::N }>],
        optimal_specimen: Genome<{ Self::N }>,
    ) -> bool
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        let (algo_type, encoding) = config.ty;
        let optimal = evaluate_phenotype(algo_type, encoding, optimal_specimen);
        final_population.iter().any(|genome| {
            let current = evaluate_phenotype(algo_type, encoding, genome.genome);
            N64::abs(current.pheonotype - optimal.pheonotype) <= N64::from(0.01)
                && N64::abs(current.fitness - optimal.fitness) <= N64::from(0.01)
        })
    }

    fn is_success_unconverged(
        _: &AlgoConfig<Self>,
        _: &[EvaluatedGenome<{ Self::N }>],
        _: Genome<{ Self::N }>,
    ) -> bool
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        false
    }
}

impl SuccessFamily for G100 {
    fn is_success_converged(
        config: &AlgoConfig<Self>,
        final_population: &[EvaluatedGenome<{ Self::N }>],
        optimal_specimen: Genome<{ Self::N }>,
    ) -> bool
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        match config.mutation_rate {
            None => final_population
                .iter()
                .all(|genome| genome.genome == optimal_specimen),
            Some(_) => {
                let optimal_specimen_count = final_population
                    .iter()
                    .filter(|genome| genome.genome == optimal_specimen)
                    .count();
                optimal_specimen_count > (final_population.len() * 9 / 10)
            }
        }
    }

    fn is_success_unconverged(
        config: &AlgoConfig<Self>,
        final_population: &[EvaluatedGenome<{ Self::N }>],
        optimal_specimen: Genome<{ Self::N }>,
    ) -> bool
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        config.mutation_rate.is_none()
            && final_population
                .iter()
                .all(|genome| genome.genome == optimal_specimen)
    }
}

pub struct OptimumlessStateEncoder {
    iterations: Vec<OptimumlessIterationStats>,
    populations: ArrayVec<PopulationStats, 5>,
}

impl OptimumlessStateEncoder {
    pub fn record_run_stat<const N: usize>(
        &mut self,
        population: &[EvaluatedGenome<N>],
        selection_stats: SelectionStats,
    ) -> N64
    where
        [(); bitvec::mem::elts::<u16>(N)]:,
    {
        let avg_fitness = avg_fitness(population);

        self.iterations.push(OptimumlessIterationStats {
            avg_fitness,
            best_fitness: population
                .iter()
                .map(|genome| genome.fitness)
                .max()
                .expect("At least one iteration to be recorded"),
            fitness_std_dev: std_dev_by(population, |genome| genome.fitness),
            rr: selection_stats.rr,
            theta: selection_stats.theta,
        });
        return avg_fitness;
    }

    pub fn record_population<const N: usize>(&mut self, population: &[EvaluatedGenome<N>])
    where
        [(); bitvec::mem::elts::<u16>(N)]:,
    {
        if self.populations.is_full() {
            return;
        }
        self.populations.push(PopulationStats {
            fitness: population.iter().map(|g| g.fitness).collect(),
            ones_count: population.iter().map(|g| g.genome.count_ones()).collect(),
            phenotype: None,
        });
    }

    pub fn finish_converged<const N: usize>(
        self,
        final_population: &[EvaluatedGenome<N>],
    ) -> OptimumlessRunStats
    where
        [(); bitvec::mem::elts::<u16>(N)]:,
    {
        self.finish(final_population, true, true)
    }

    pub fn finish_unconverged<const N: usize>(
        self,
        final_population: &[EvaluatedGenome<N>],
    ) -> OptimumlessRunStats
    where
        [(); bitvec::mem::elts::<u16>(N)]:,
    {
        self.finish(final_population, false, false)
    }

    fn finish<const N: usize>(
        self,
        final_population: &[EvaluatedGenome<N>],
        success: bool,
        converged: bool,
    ) -> OptimumlessRunStats
    where
        [(); bitvec::mem::elts::<u16>(N)]:,
    {
        OptimumlessRunStats {
            success,
            converged,
            min_rr: iteration_min(&self.iterations, |i| i.rr),
            max_rr: iteration_max(&self.iterations, |i| i.rr),
            avg_rr: avg_by(&self.iterations, |i| i.rr),
            min_theta: iteration_min(&self.iterations, |i| i.theta),
            max_theta: iteration_max(&self.iterations, |i| i.theta),
            avg_theta: avg_by(&self.iterations, |i| i.theta),

            iterations: self.iterations,
            starting_population: self.populations,
            final_population: PopulationStats {
                fitness: final_population.iter().map(|g| g.fitness).collect(),
                ones_count: final_population
                    .iter()
                    .map(|g| g.genome.count_ones())
                    .collect(),
                phenotype: None,
            },
        }
    }
}

impl<'a, F: SuccessFamily + EvaluateFamily> StatEncoder<'a, F>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    pub fn new(config: &'a AlgoConfig<F>) -> Self {
        match config.optimal_specimen {
            Some(optimal_specimen) => Self::WithOptimum(StateEncoderWithOptimum {
                config,
                optimal_specimen,
                iterations: Vec::new(),
                prev_iteration_optimal_specimens: 0,
                populations: ArrayVec::new(),
            }),
            None => Self::Optimumless(OptimumlessStateEncoder {
                iterations: Vec::new(),
                populations: ArrayVec::new(),
            }),
        }
    }

    pub fn record_run_stat(
        &mut self,
        population: &[EvaluatedGenome<{ F::N }>],
        selection_stats: SelectionStats,
    ) -> N64 {
        match self {
            Self::WithOptimum(stats) => stats.record_run_stat(population, selection_stats),
            Self::Optimumless(stats) => stats.record_run_stat(population, selection_stats),
        }
    }

    pub fn finish_converged(self, final_population: &[EvaluatedGenome<{ F::N }>]) -> RunStats {
        use OptimumDisabiguity::*;
        match self {
            WithOptimum(stats) => WithOptimum(stats.finish_converged(final_population)),
            Optimumless(stats) => Optimumless(stats.finish_converged(final_population)),
        }
    }

    pub fn finish_unconverged(self, final_population: &[EvaluatedGenome<{ F::N }>]) -> RunStats {
        use OptimumDisabiguity::*;
        match self {
            WithOptimum(stats) => WithOptimum(stats.finish_unconverged(final_population)),
            Optimumless(stats) => Optimumless(stats.finish_unconverged(final_population)),
        }
    }

    pub fn record_population(&mut self, population: &[EvaluatedGenome<{ F::N }>]) {
        match self {
            Self::WithOptimum(stats) => stats.record_population(population),
            Self::Optimumless(stats) => stats.record_population(population),
        }
    }
}
