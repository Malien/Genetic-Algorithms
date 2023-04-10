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
    }, AlgoDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfigKey<T> {
    pub algo_type: T,
    pub population_size: usize,
    pub apply_crossover: bool,
    pub apply_mutation: bool,
    pub selection: Selection,
}

impl<T: AlgoDescriptor> ConfigKey<T> {
    fn to_path(&self) -> PathBuf {
        let mut path = PathBuf::new();
        path.push("data");

        path.push(self.algo_type.name());
        if let Some(args) = self.algo_type.args() {
            path.push(args);
        }

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

    fn selection_name(&self) -> &'static str {
        match self.selection {
            Selection::StochasticTournament { .. } => "stochastic-tournament",
        }
    }

    fn tournament_replacement(&self) -> &'static str {
        match self.selection {
            Selection::StochasticTournament {
                replacement: TournamentReplacement::With,
                ..
            } => "with",
            Selection::StochasticTournament {
                replacement: TournamentReplacement::Without,
                ..
            } => "without",
        }
    }

    fn selection_prob(&self) -> String {
        match self.selection {
            Selection::StochasticTournament { prob, .. } => format!("{:.1}", prob),
        }
    }
}

pub fn create_config_writer() -> io::Result<csv::Writer<impl std::io::Write>> {
    std::fs::create_dir_all("data")?;
    let file = File::create("data/stats.csv")?;
    let writer = BufWriter::new(file);
    let mut writer = csv::Writer::from_writer(writer);

    writer.write_record([
        "algo_name",
        "algo_args",
        "population_size",
        "apply_crossover",
        "apply_mutation",
        "selection_method",
        "tournament_replacement",
        "selection_prob",

        "success_percent",
        "min_iteration_count",
        "max_iteration_count",
        "avg_iteration_count",
        "iteration_count_std_dev",
        "min_min_selection_intensity_idx",
        "min_min_selection_intensity",
        "max_max_selection_intensity_idx",
        "max_max_selection_intensity",
        "avg_min_selection_intensity",
        "avg_max_selection_intensity",
        "avg_selection_intensity",
        "min_selection_intensity_std_dev",
        "max_selection_intensity_std_dev",
        "avg_selection_intensity_std_dev",
        "min_early_growth_rate",
        "max_early_growth_rate",
        "avg_early_growth_rate",
        "min_late_growth_rate",
        "max_late_growth_rate",
        "avg_late_growth_rate",
        "min_avg_growth_rate",
        "max_avg_growth_rate",
        "avg_avg_growth_rate",
        "min_min_rr_idx",
        "min_min_rr",
        "max_max_rr_idx",
        "max_max_rr",
        "avg_min_rr",
        "avg_max_rr",
        "avg_avg_rr",
        "min_rr_std_dev",
        "max_rr_std_dev",
        "avg_rr_std_dev",
        "min_min_theta_idx",
        "min_min_theta",
        "max_max_theta_idx",
        "max_max_theta",
        "avg_min_theta",
        "avg_max_theta",
        "avg_avg_theta",
        "min_theta_std_dev",
        "max_theta_std_dev",
        "avg_theta_std_dev",
    ])?;

    Ok(writer)
}

pub fn append_config<T: AlgoDescriptor>(
    writer: &mut csv::Writer<impl std::io::Write>,
    key: ConfigKey<T>,
    stats: &ConfigStats,
) -> io::Result<()> {
    use OptimumDisabiguity::*;

    match stats {
        WithOptimum(stats) => append_config_with_optimum(writer, key, stats),
        Optimumless(stats) => append_config_optimumless(writer, key, stats),
    }?;

    Ok(())
}

fn append_config_with_optimum<T: AlgoDescriptor>(
    writer: &mut csv::Writer<impl std::io::Write>,
    key: ConfigKey<T>,
    stats: &ConfigStatsWithOptimum,
) -> io::Result<()> {
    writer.write_record([
        key.algo_type.name(),
        &key.algo_type.args().unwrap_or_default(),
        &key.population_size.to_string(),
        &key.apply_crossover.to_string(),
        &key.apply_mutation.to_string(),
        key.selection_name(),
        key.tournament_replacement(),
        &key.selection_prob(),

        &stats.success_percent.to_string(),
        &stats.min_iteration_count.map(|v| v.to_string()).unwrap_or_default(),
        &stats.max_iteration_count.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_iteration_count.map(|v| v.to_string()).unwrap_or_default(),
        &stats.iteration_count_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_min_selection_intensity.0.to_string(),
        &stats.min_min_selection_intensity.1.to_string(),
        &stats.max_max_selection_intensity.0.to_string(),
        &stats.max_max_selection_intensity.1.to_string(),
        &stats.avg_min_selection_intensity.to_string(),
        &stats.avg_max_selection_intensity.to_string(),
        &stats.avg_selection_intensity.to_string(),
        &stats.min_selection_intensity_std_dev.to_string(),
        &stats.max_selection_intensity_std_dev.to_string(),
        &stats.avg_selection_intensity_std_dev.to_string(),
        &stats.min_early_growth_rate.to_string(),
        &stats.max_early_growth_rate.to_string(),
        &stats.avg_early_growth_rate.to_string(),
        &stats.min_late_growth_rate.map(|v| v.to_string()).unwrap_or_default(),
        &stats.max_late_growth_rate.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_late_growth_rate.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_avg_growth_rate.to_string(),
        &stats.max_avg_growth_rate.to_string(),
        &stats.avg_avg_growth_rate.to_string(),
        &stats.min_min_rr.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.min_min_rr.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.max_max_rr.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.max_max_rr.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.avg_min_rr.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_max_rr.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_avg_rr.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_rr_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.max_rr_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_rr_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_min_theta.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.min_min_theta.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.max_max_theta.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.max_max_theta.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.avg_min_theta.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_max_theta.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_avg_theta.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_theta_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.max_theta_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_theta_std_dev.map(|v| v.to_string()).unwrap_or_default(),
    ])?;

    Ok(())
}

fn append_config_optimumless<T: AlgoDescriptor>(
    writer: &mut csv::Writer<impl std::io::Write>,
    key: ConfigKey<T>,
    stats: &OptimumlessConfigStats,
) -> io::Result<()> {
    writer.write_record([
        key.algo_type.name(),
        &key.algo_type.args().unwrap_or_default(),
        &key.population_size.to_string(),
        &key.apply_crossover.to_string(),
        &key.apply_mutation.to_string(),
        key.selection_name(),
        key.tournament_replacement(),
        &key.selection_prob(),

        &stats.success_percent.to_string(),
        &stats.min_iteration_count.map(|v| v.to_string()).unwrap_or_default(),
        &stats.max_iteration_count.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_iteration_count.map(|v| v.to_string()).unwrap_or_default(),
        &stats.iteration_count_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        &stats.min_min_rr.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.min_min_rr.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.max_max_rr.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.max_max_rr.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.avg_min_rr.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_max_rr.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_avg_rr.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_rr_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.max_rr_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_rr_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_min_theta.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.min_min_theta.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.max_max_theta.map(|v| v.0.to_string()).unwrap_or_default(),
        &stats.max_max_theta.map(|v| v.1.to_string()).unwrap_or_default(),
        &stats.avg_min_theta.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_max_theta.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_avg_theta.map(|v| v.to_string()).unwrap_or_default(),
        &stats.min_theta_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.max_theta_std_dev.map(|v| v.to_string()).unwrap_or_default(),
        &stats.avg_theta_std_dev.map(|v| v.to_string()).unwrap_or_default(),
    ])?;

    Ok(())
}

pub fn write_stats<T: AlgoDescriptor>(key: ConfigKey<T>, stats: &ConfigStats) -> io::Result<()> {
    use OptimumDisabiguity::*;
    let path = key.to_path();

    match stats {
        WithOptimum(stats) => write_with_optimum(&path, stats),
        Optimumless(stats) => write_optimumless(&path, stats),
    }
}

fn write_optimumless(path: &Path, stats: &OptimumlessConfigStats) -> io::Result<()> {
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

    for (run, n) in stats.runs.iter().zip(1..) {
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

fn write_with_optimum(path: &Path, stats: &ConfigStatsWithOptimum) -> io::Result<()> {
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

    for (run, n) in stats.runs.iter().zip(1..) {
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

fn write_run_with_optimum(path: &Path, run: &RunStatsWithOptimum) -> io::Result<()> {
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

    for iteration in &run.iterations {
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

    for (idx, population) in run.starting_population.iter().enumerate() {
        write_iteration_population(&path.join(format!("{idx}")), population.iter())?;
    }

    write_iteration_population(&path.join("final"), run.final_population.iter())?;

    Ok(())
}

fn write_run_optimumless(path: &Path, run: &OptimumlessRunStats) -> io::Result<()> {
    std::fs::create_dir_all(path)?;

    let file = File::create(path.join("iterations.csv"))?;
    let writer = BufWriter::new(file);
    let mut writer = csv::Writer::from_writer(writer);

    writer.write_record(["avg_fitness", "rr", "theta"])?;

    for iteration in &run.iterations {
        writer.write_record([
            iteration.avg_fitness.to_string(),
            iteration.rr.to_string(),
            iteration.theta.to_string(),
        ])?;
    }

    for (idx, population) in run.starting_population.iter().enumerate() {
        write_iteration_population(&path.join(format!("{idx}")), population)?;
    }

    write_iteration_population(&path.join("final"), run.final_population.iter())?;

    Ok(())
}

fn write_iteration_population<'a>(
    path: &Path,
    population: impl IntoIterator<Item = &'a PopulationStats>,
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
