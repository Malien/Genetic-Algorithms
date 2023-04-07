use decorum::N64;
use num_traits::real::Real;

use crate::{
    avg_fitness,
    evaluation::{evaluate_phenotype, AlgoType},
    AlgoConfig, EvaluatedGenome, Genome,
};

#[derive(Debug, Clone, Copy)]
pub struct IterationStats {
    selection_intensity: N64,
    growth_rate: f64,
    optimal_specimen_count: usize,
    avg_fitness: N64,
    rr: f64,
    theta: f64,
    selection_diff: N64,
}

#[derive(Debug, Clone, Copy)]
pub struct OptimumlessIterationStats {
    selection_intensity: N64,
    avg_fitness: N64,
    rr: f64,
    theta: f64,
    selection_diff: N64,
}

#[derive(Debug, Clone)]
pub struct RunStatsWithOptimum {
    iterations: Vec<IterationStats>,
    best_fitness: N64,
    avg_fitness: N64,
    success: bool,
}

#[derive(Debug, Clone)]
pub struct OptimumlessRunStats {
    iterations: Vec<OptimumlessIterationStats>,
    best_fitness: N64,
    avg_fitness: N64,
    success: bool,
}

#[derive(Debug, Clone)]
pub enum OptimumDisabiguity<With, Without> {
    WithOptimum(With),
    Optimumless(Without),
}

pub type RunStats = OptimumDisabiguity<RunStatsWithOptimum, OptimumlessRunStats>;

pub type StatEncoder<'a> =
    OptimumDisabiguity<StateEncoderWithOptimum<'a>, OptimumlessStateEncoder>;

pub struct StateEncoderWithOptimum<'a> {
    config: &'a AlgoConfig,
    optimal_specimen: &'a Genome,
    iterations: Vec<IterationStats>,
    prev_iteration_optimal_specimens: usize,
}

impl StateEncoderWithOptimum<'_> {
    pub fn record_run_stat(
        &mut self,
        pre_selection_fitness: N64,
        population: &[EvaluatedGenome],
        unique_specimens_selected: usize,
    ) -> N64 {
        let optimal_specimen_count = self
            .config
            .optimal_specimen
            .as_ref()
            .map(|optimal_specimen| {
                population
                    .iter()
                    .filter(|genome| &genome.genome == optimal_specimen)
                    .count()
            })
            .unwrap_or(0);
        let population_size = population.len() as f64;
        let avg_fitness = avg_fitness(population);
        let std_dev = population
            .iter()
            .map(|g| N64::powi(g.fitness - avg_fitness, 2))
            .sum::<N64>()
            / population_size;

        self.iterations.push(IterationStats {
            selection_intensity: (avg_fitness - pre_selection_fitness) / std_dev,
            selection_diff: avg_fitness - pre_selection_fitness,
            growth_rate: optimal_specimen_count as f64
                / self.prev_iteration_optimal_specimens as f64,
            optimal_specimen_count,
            avg_fitness,
            rr: unique_specimens_selected as f64 / population_size,
            theta: (population_size - unique_specimens_selected as f64) / population_size,
        });
        self.prev_iteration_optimal_specimens = optimal_specimen_count;
        return avg_fitness;
    }

    pub fn finish_converged(self, final_population: &[EvaluatedGenome]) -> RunStatsWithOptimum {
        let success = match (self.config.mutation_rate, self.config.ty) {
            (None, AlgoType::BinaryOnly(_)) => final_population
                .iter()
                .all(|genome| &genome.genome == self.optimal_specimen),
            (Some(_), AlgoType::BinaryOnly(_)) => {
                let optimal_specimen_count = final_population
                    .iter()
                    .filter(|genome| &genome.genome == self.optimal_specimen)
                    .count();
                optimal_specimen_count > (final_population.len() * 9 / 10)
            }
            (_, AlgoType::HasPhoenotype(algo_type, encoding)) => {
                let optimal = evaluate_phenotype(algo_type, encoding, &self.optimal_specimen);
                final_population.iter().any(|genome| {
                    let current = evaluate_phenotype(algo_type, encoding, &genome.genome);
                    N64::abs(current.pheonotype - optimal.pheonotype) <= N64::from(0.01)
                        && N64::abs(current.fitness - optimal.fitness) <= N64::from(0.01)
                })
            }
        };
        self.finish(final_population, success)
    }

    pub fn finish_unconverged(self, final_population: &[EvaluatedGenome]) -> RunStatsWithOptimum {
        let success = self.config.mutation_rate.is_none()
            && matches!(self.config.ty, AlgoType::BinaryOnly(_))
            && final_population
                .iter()
                .all(|genome| &genome.genome == self.optimal_specimen);
        self.finish(final_population, success)
    }

    fn finish(self, final_population: &[EvaluatedGenome], success: bool) -> RunStatsWithOptimum {
        RunStatsWithOptimum {
            iterations: self.iterations,
            best_fitness: final_population
                .iter()
                .map(|g| g.fitness)
                .max()
                .expect("Population is not empty"),
            avg_fitness: avg_fitness(&final_population),
            success,
        }
    }
}

pub struct OptimumlessStateEncoder {
    iterations: Vec<OptimumlessIterationStats>,
}

impl OptimumlessStateEncoder {
    pub fn record_run_stat(
        &mut self,
        pre_selection_fitness: N64,
        population: &[EvaluatedGenome],
        unique_specimens_selected: usize,
    ) -> N64 {
        let population_size = population.len() as f64;
        let avg_fitness = avg_fitness(population);
        let std_dev = population
            .iter()
            .map(|g| N64::powi(g.fitness - avg_fitness, 2))
            .sum::<N64>()
            / population_size;

        self.iterations.push(OptimumlessIterationStats {
            selection_intensity: (avg_fitness - pre_selection_fitness) / std_dev,
            selection_diff: avg_fitness - pre_selection_fitness,
            avg_fitness,
            rr: unique_specimens_selected as f64 / population_size,
            theta: (population_size - unique_specimens_selected as f64) / population_size,
        });
        return avg_fitness;
    }

    pub fn finish_converged(self, final_population: &[EvaluatedGenome]) -> OptimumlessRunStats {
        self.finish(final_population, true)
    }

    pub fn finish_unconverged(self, final_population: &[EvaluatedGenome]) -> OptimumlessRunStats {
        self.finish(final_population, false)
    }

    fn finish(self, final_population: &[EvaluatedGenome], success: bool) -> OptimumlessRunStats {
        OptimumlessRunStats {
            iterations: self.iterations,
            best_fitness: final_population
                .iter()
                .map(|g| g.fitness)
                .max()
                .expect("Population is not empty"),
            avg_fitness: avg_fitness(&final_population),
            success,
        }
    }
}

impl<'a> StatEncoder<'a> {
    pub fn new(config: &'a AlgoConfig) -> Self {
        match config.optimal_specimen {
            Some(ref optimal_specimen) => Self::WithOptimum(StateEncoderWithOptimum {
                config,
                optimal_specimen,
                iterations: Vec::new(),
                prev_iteration_optimal_specimens: 0,
            }),
            None => Self::Optimumless(OptimumlessStateEncoder {
                iterations: Vec::new(),
            }),
        }
    }

    pub fn record_run_stat(
        &mut self,
        pre_selection_fitness: N64,
        population: &[EvaluatedGenome],
        unique_specimens_selected: usize,
    ) -> N64 {
        match self {
            Self::WithOptimum(stats) => {
                stats.record_run_stat(pre_selection_fitness, population, unique_specimens_selected)
            }
            Self::Optimumless(stats) => {
                stats.record_run_stat(pre_selection_fitness, population, unique_specimens_selected)
            }
        }
    }

    pub fn finish_converged(self, final_population: &[EvaluatedGenome]) -> RunStats {
        use OptimumDisabiguity::*;
        match self {
            WithOptimum(stats) => WithOptimum(stats.finish_converged(final_population)),
            Optimumless(stats) => Optimumless(stats.finish_converged(final_population)),
        }
    }

    pub fn finish_unconverged(self, final_population: &[EvaluatedGenome]) -> RunStats {
        use OptimumDisabiguity::*;
        match self {
            WithOptimum(stats) => WithOptimum(stats.finish_unconverged(final_population)),
            Optimumless(stats) => Optimumless(stats.finish_unconverged(final_population)),
        }
    }
}
