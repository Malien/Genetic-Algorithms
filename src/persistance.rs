use std::{
    fs::File,
    io::{self, BufWriter},
    path::{Path, PathBuf},
};

use crate::{
    selection::{Selection, TournamentReplacement},
    stats::{
        ConfigStats, ConfigStatsWithOptimum, OptimumDisabiguity, OptimumlessConfigStats,
        OptimumlessRunStats, PopulationStats, RunStatsWithOptimum,
    },
    ToPath,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfigKey<T> {
    pub algo_type: T,
    pub population_size: usize,
    pub apply_crossover: bool,
    pub apply_mutation: bool,
    pub selection: Selection,
}

impl<T: ToPath> ConfigKey<T> {
    fn to_path(&self) -> PathBuf {
        let mut path = PathBuf::new();
        path.push("data");

        path.push(self.algo_type.to_path());

        path.push(if self.apply_crossover {
            "crossover"
        } else {
            "no-crossover"
        });
        path.push(if self.apply_mutation {
            "mutation"
        } else {
            "no-mutation"
        });
        path.push(format!("{}", self.population_size));
        match self.selection {
            Selection::StochasticTournament {
                prob,
                replacement: TournamentReplacement::With,
            } => {
                path.push("stochastic-tournament");
                path.push("with-replacement");
                path.push(format!("{:.1}", prob));
            }
            Selection::StochasticTournament {
                prob,
                replacement: TournamentReplacement::Without,
            } => {
                path.push("stochastic-tournament");
                path.push("without-replacement");
                path.push(format!("{:.1}", prob));
            }
        };
        path
    }
}

pub fn write_stats<T: ToPath>(key: ConfigKey<T>, stats: ConfigStats) -> io::Result<()> {
    use OptimumDisabiguity::*;
    let path = key.to_path();

    match stats {
        WithOptimum(stats) => write_with_optimum(&path, stats),
        Optimumless(stats) => write_optimumless(&path, stats),
    }
}

fn write_optimumless(path: &Path, stats: OptimumlessConfigStats) -> io::Result<()> {
    std::fs::create_dir_all(path)?;

    let file = File::create(path.join("runs.csv"))?;
    let writer = BufWriter::new(file);
    let mut writer = csv::Writer::from_writer(writer);

    writer.write_record([
        "iteration_count",
        "success",
        "min_rr_idx",
        "min_rr",
        "max_rr_idx",
        "max_rr",
        "avg_rr",
        "min_theta_idx",
        "min_theta",
        "max_theta_idx",
        "max_theta",
        "avg_theta",
    ])?;

    for (run, n) in stats.runs.into_iter().zip(1..) {
        writer.write_record([
            run.iterations.len().to_string(),
            run.success.to_string(),
            run.min_rr.0.to_string(),
            run.min_rr.1.to_string(),
            run.max_rr.0.to_string(),
            run.max_rr.1.to_string(),
            run.avg_rr.to_string(),
            run.min_theta.0.to_string(),
            run.min_theta.1.to_string(),
            run.max_theta.0.to_string(),
            run.max_theta.1.to_string(),
            run.avg_theta.to_string(),
        ])?;

        write_run_optimumless(&path.join(format!("{n}")), run)?;
    }

    Ok(())
}

fn write_with_optimum(path: &Path, stats: ConfigStatsWithOptimum) -> io::Result<()> {
    std::fs::create_dir_all(path)?;

    let file = File::create(path.join("runs.csv"))?;
    let writer = BufWriter::new(file);
    let mut writer = csv::Writer::from_writer(writer);

    writer.write_record([
        "iteration_count",
        "best_fitness",
        "avg_fitness",
        "success",
        "min_selection_intensity_idx",
        "min_selection_intensity",
        "max_selection_intensity_idx",
        "max_selection_intensity",
        "avg_selection_intensity",
        "early_growth_rate",
        "avg_growth_rate",
        "late_growth_rate_idx",
        "late_growth_rate",
        "min_rr_idx",
        "min_rr",
        "max_rr_idx",
        "max_rr",
        "avg_rr",
        "min_theta_idx",
        "min_theta",
        "max_theta_idx",
        "max_theta",
        "avg_theta",
        "min_selection_diff_idx",
        "min_selection_diff",
        "max_selection_diff_idx",
        "max_selection_diff",
        "avg_selection_diff",
    ])?;

    for (run, n) in stats.runs.into_iter().zip(1..) {
        writer.write_record([
            run.iterations.len().to_string(),
            run.best_fitness.to_string(),
            run.avg_fitness.to_string(),
            run.success.to_string(),
            run.min_selection_intensity.0.to_string(),
            run.min_selection_intensity.1.to_string(),
            run.max_selection_intensity.0.to_string(),
            run.max_selection_intensity.1.to_string(),
            run.avg_selection_intensity.to_string(),
            run.early_growth_rate.to_string(),
            run.avg_growth_rate.to_string(),
            run.late_growth_rate
                .map(|(idx, _)| idx.to_string())
                .unwrap_or_default(),
            run.late_growth_rate
                .map(|(_, v)| v.to_string())
                .unwrap_or_default(),
            run.min_rr.0.to_string(),
            run.min_rr.1.to_string(),
            run.max_rr.0.to_string(),
            run.max_rr.1.to_string(),
            run.avg_rr.to_string(),
            run.min_theta.0.to_string(),
            run.min_theta.1.to_string(),
            run.max_theta.0.to_string(),
            run.max_theta.1.to_string(),
            run.avg_theta.to_string(),
            run.min_selection_diff.0.to_string(),
            run.min_selection_diff.1.to_string(),
            run.max_selection_diff.0.to_string(),
            run.max_selection_diff.1.to_string(),
            run.avg_selection_diff.to_string(),
        ])?;

        write_run_with_optimum(&path.join(format!("{n}")), run)?;
    }

    Ok(())
}

fn write_run_with_optimum(path: &Path, run: RunStatsWithOptimum) -> io::Result<()> {
    std::fs::create_dir_all(path)?;

    let file = File::create(path.join("iterations.csv"))?;
    let writer = BufWriter::new(file);
    let mut writer = csv::Writer::from_writer(writer);

    writer.write_record([
        "selection_intensity",
        "growth_rate",
        "optimal_specimen_count",
        "avg_fitness",
        "rr",
        "theta",
        "selection_diff",
    ])?;

    for iteration in run.iterations {
        writer.write_record([
            iteration.selection_intensity.to_string(),
            iteration.growth_rate.to_string(),
            iteration.optimal_specimen_count.to_string(),
            iteration.avg_fitness.to_string(),
            iteration.rr.to_string(),
            iteration.theta.to_string(),
            iteration.selection_diff.to_string(),
        ])?;
    }

    for (idx, population) in run.starting_population.into_iter().enumerate() {
        write_iteration_population(&path.join(format!("{idx}")), population)?;
    }

    write_iteration_population(&path.join("final"), run.final_population)?;

    Ok(())
}

fn write_run_optimumless(path: &Path, run: OptimumlessRunStats) -> io::Result<()> {
    std::fs::create_dir_all(path)?;

    let file = File::create(path.join("iterations.csv"))?;
    let writer = BufWriter::new(file);
    let mut writer = csv::Writer::from_writer(writer);

    writer.write_record(["avg_fitness", "rr", "theta"])?;

    for iteration in run.iterations {
        writer.write_record([
            iteration.avg_fitness.to_string(),
            iteration.rr.to_string(),
            iteration.theta.to_string(),
        ])?;
    }

    for (idx, population) in run.starting_population.into_iter().enumerate() {
        write_iteration_population(&path.join(format!("{idx}")), population)?;
    }

    write_iteration_population(&path.join("final"), run.final_population)?;

    Ok(())
}

fn write_iteration_population(
    path: &Path,
    population: impl IntoIterator<Item = PopulationStats>,
) -> io::Result<()> {
    std::fs::create_dir_all(path)?;

    let file = File::create(path.join("population.csv"))?;
    let writer = BufWriter::new(file);
    let mut writer = csv::Writer::from_writer(writer);

    writer.write_record(["fitness", "phenotype", "ones_count"])?;

    for population in population {
        writer.write_record([
            population.fitness.to_string(),
            population
                .phenotype
                .map(|v| v.to_string())
                .unwrap_or_default(),
            population.ones_count.to_string(),
        ])?;
    }

    Ok(())
}
