use arrayvec::ArrayVec;
use decorum::N64;
use num_traits::real::Real;
use plotters::{
    backend::{BitMapBackend, PixelFormat, RGBPixel},
    chart::ChartBuilder,
    coord::{combinators::IntoLinspace, Shift},
    drawing::IntoDrawingArea,
    prelude::PathElement,
    series::{Histogram, LineSeries},
    style::{colors, Color, IntoFont},
};

use crate::stats::{
    ConfigStats, OptimumDisabiguity, OptimumlessRunStats, PopulationStats, RunStatsWithOptimum,
};
use crate::{BinaryAlgo, GenomeEncoding, PheonotypeAlgo};

const GRAPH_WIDTH: u32 = 600;
const GRAPH_HEIGHT: u32 = 400;

type DrawingArea<'a> = plotters::drawing::DrawingArea<BitMapBackend<'a, RGBPixel>, Shift>;

pub struct RunGraphs {
    pub starting_population: ArrayVec<PopulationGraphs, 5>,
    pub final_population: PopulationGraphs,
    pub avg_fitness: Option<Vec<u8>>,
    pub best_fitness: Option<Vec<u8>>,
    pub fitness_std_dev: Option<Vec<u8>>,
    pub rr_and_theta: Option<Vec<u8>>,
    pub selection_intensity: Option<Vec<u8>>,
    pub selection_diff: Option<Vec<u8>>,
    pub optimal_specimen_count: Option<Vec<u8>>,
    pub growth_rate: Option<Vec<u8>>,
    pub selection_intensity_and_diff: Option<Vec<u8>>,
}

pub struct PopulationGraphs {
    pub fitness: Option<Vec<u8>>,
    pub phenotype: Option<Vec<u8>>,
    pub ones_count: Vec<u8>,
}

fn graph_image(configure: impl FnOnce(&DrawingArea) -> eyre::Result<()>) -> eyre::Result<Vec<u8>> {
    let mut buf = vec![0; GRAPH_WIDTH as usize * GRAPH_HEIGHT as usize * RGBPixel::PIXEL_SIZE];
    let root =
        BitMapBackend::<RGBPixel>::with_buffer_and_format(&mut buf, (GRAPH_WIDTH, GRAPH_HEIGHT))?
            .into_drawing_area();
    root.fill(&colors::WHITE)?;

    configure(&root)?;

    root.present()?;
    drop(root);

    use image::codecs::png::*;
    use image::ImageEncoder;
    let mut encoded_buf = Vec::with_capacity(16 * 1024);
    let encoder = PngEncoder::new_with_quality(
        &mut encoded_buf,
        CompressionType::Default,
        FilterType::NoFilter,
    );
    encoder.write_image(
        &buf,
        GRAPH_WIDTH as u32,
        GRAPH_HEIGHT as u32,
        image::ColorType::Rgb8,
    )?;

    Ok(encoded_buf)
}

fn population_ones_count_graph(one_counts: &[usize]) -> eyre::Result<Vec<u8>> {
    let max = one_counts.iter().max().unwrap();
    let bucket_count = hist_ceiling_usize((0, *max), one_counts);
    graph_image(|root| {
        let mut chart = ChartBuilder::on(&root)
            .caption("Amount of ones in genome", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(20)
            .build_cartesian_2d(0..max + 1, 0..bucket_count)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(colors::BLUE.filled())
                .margin(10)
                .data(one_counts.iter().map(|&count| (count, 1))),
        )?;

        Ok(())
    })
}

fn hist_ceiling(bounds: Bounds, values: &[N64]) -> u32 {
    let len = f64::ceil((bounds.max_f() - bounds.min_f()) / bounds.step) as usize;
    let mut buckets = vec![0; len + 1];
    for &value in values {
        let index = N64::round((value - bounds.min) / bounds.step);
        buckets[f64::from(index) as usize] += 1;
    }
    buckets.into_iter().max().unwrap()
}

fn hist_ceiling_usize((min, max): (usize, usize), values: &[usize]) -> usize {
    let len = max - min;
    let mut buckets = vec![0; len + 1];
    for value in values {
        buckets[value - min] += 1;
    }
    buckets.into_iter().max().unwrap()
}

fn population_phenotype_graph(bounds: Bounds, phenotypes: &[N64]) -> eyre::Result<Vec<u8>> {
    graph_image(|root| {
        let mut chart = ChartBuilder::on(&root)
            .caption("Phenotype distribution", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(20)
            .build_cartesian_2d(
                (bounds.min_f()..bounds.max_f())
                    .step(bounds.step)
                    .use_round(),
                0..hist_ceiling(bounds, phenotypes),
            )?;

        chart.configure_mesh().draw()?;

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(colors::BLUE.filled())
                .margin(2)
                .data(phenotypes.iter().map(|&phenotype| (phenotype, 1))),
        )?;

        Ok(())
    })
}

fn population_fitness_graph(bounds: Bounds, fitnesses: &[N64]) -> eyre::Result<Vec<u8>> {
    graph_image(|root| {
        let mut chart = ChartBuilder::on(&root)
            .caption("Fitness distribution", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(20)
            .build_cartesian_2d(
                (bounds.min_f()..bounds.max_f() + bounds.step)
                    .step(bounds.step)
                    .use_round(),
                0..hist_ceiling(bounds, fitnesses),
            )?;

        chart.configure_mesh().draw()?;

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(colors::BLUE.filled())
                .margin(2)
                .data(fitnesses.iter().map(|&fitness| (fitness, 1))),
        )?;

        Ok(())
    })
}

fn minmax_iter<T: Ord + Copy>(iter: impl IntoIterator<Item = T>) -> Option<(T, T)> {
    let mut iter = iter.into_iter();
    let mut min = iter.next()?;
    let mut max = min;
    for value in iter {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    Some((min, max))
}

fn just_plot_floats<I>(name: &str, (min, max): (N64, N64), iter: I) -> eyre::Result<Vec<u8>>
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator<Item = N64>,
{
    let iter = iter.into_iter();
    graph_image(|root| {
        let mut chart = ChartBuilder::on(&root)
            .caption(name, ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..(iter.len() - 1), f64::from(min)..f64::from(max))?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            iter.map(f64::from).enumerate(),
            &colors::RED,
        ))?;

        Ok(())
    })
}

fn just_plot_ints<I>(name: &str, (min, max): (usize, usize), iter: I) -> eyre::Result<Vec<u8>>
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator<Item = usize>,
{
    let iter = iter.into_iter();
    graph_image(|root| {
        let mut chart = ChartBuilder::on(&root)
            .caption(name, ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..(iter.len() - 1), min..max + 1)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(iter.enumerate(), &colors::RED))?;

        Ok(())
    })
}

fn plot_double(
    (min, max): (N64, N64),
    (a, a_name): (impl ExactSizeIterator<Item = N64> + Clone, &str),
    (b, b_name): (impl ExactSizeIterator<Item = N64> + Clone, &str),
) -> eyre::Result<Vec<u8>> {
    assert!(a.len() == b.len());

    graph_image(|root| {
        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("{a_name} and {b_name}"),
                ("sans-serif", 50).into_font(),
            )
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..(a.len() - 1), f64::from(min)..f64::from(max))?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(a.map(f64::from).enumerate(), &colors::RED))?
            .label(a_name)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &colors::RED));
        chart
            .draw_series(LineSeries::new(b.map(f64::from).enumerate(), &colors::BLUE))?
            .label(b_name)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &colors::BLUE));

        chart
            .configure_series_labels()
            .background_style(colors::WHITE.mix(0.8))
            .border_style(colors::BLACK)
            .draw()?;

        Ok(())
    })
}

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    min: N64,
    max: N64,
    step: f64,
}

impl Bounds {
    fn min_f(self) -> f64 {
        f64::from(self.min)
    }
    fn max_f(self) -> f64 {
        f64::from(self.max)
    }
}

impl Bounds {
    fn new(min: impl Into<N64>, max: impl Into<N64>, step: f64) -> Self {
        Self {
            min: min.into(),
            max: max.into(),
            step,
        }
    }
}

pub trait GraphDescriptor {
    fn phenotype_bounds(&self) -> Option<Bounds>;
    fn fitness_bounds(&self) -> Option<Bounds>;
}

impl GraphDescriptor for (PheonotypeAlgo, GenomeEncoding) {
    fn phenotype_bounds(&self) -> Option<Bounds> {
        Some(match self.0 {
            PheonotypeAlgo::Pow1 => Bounds::new(0.0, 10.23, 0.5),
            PheonotypeAlgo::Pow2 => Bounds::new(-5.12, 5.12, 0.5),
        })
    }

    fn fitness_bounds(&self) -> Option<Bounds> {
        Some(match self.0 {
            PheonotypeAlgo::Pow1 => Bounds::new(0.0, 10.23.powi(2), 5.0),
            PheonotypeAlgo::Pow2 => Bounds::new(0.0, 5.12.powi(2), 2.0),
        })
    }
}

impl GraphDescriptor for BinaryAlgo {
    fn phenotype_bounds(&self) -> Option<Bounds> {
        None
    }
    fn fitness_bounds(&self) -> Option<Bounds> {
        match self {
            BinaryAlgo::FHD { sigma } => Some(Bounds::new(0.0, *sigma * N64::from(100.0), 50.0)),
            BinaryAlgo::FConst => None,
        }
    }
}

pub fn draw_population_graphs(
    descriptor: &impl GraphDescriptor,
    population: &PopulationStats,
) -> eyre::Result<PopulationGraphs> {
    let phenotype_bounds = descriptor.phenotype_bounds();
    let fitness_bounds = descriptor.fitness_bounds();

    let ones_count = population_ones_count_graph(&population.ones_count)?;
    let phenotype = match (phenotype_bounds, &population.phenotype) {
        (Some(bounds), Some(phenotypes)) => Some(population_phenotype_graph(bounds, phenotypes)?),
        _ => None,
    };

    let fitness = match fitness_bounds {
        Some(bounds) => Some(population_fitness_graph(bounds, &population.fitness)?),
        _ => None,
    };

    Ok(PopulationGraphs {
        fitness,
        phenotype,
        ones_count,
    })
}

macro_rules! rewrap {
    ($opt:expr, |$inner:pat_param| $block:expr) => {
        match $opt {
            Some($inner) => Some($block),
            None => None,
        }
    };
}

pub fn draw_optimumless_run_graphs(
    descriptor: &impl GraphDescriptor,
    run: &OptimumlessRunStats,
) -> eyre::Result<RunGraphs> {
    let starting_population = run
        .starting_population
        .iter()
        .map(|p| draw_population_graphs(descriptor, p))
        .collect::<Result<_, _>>()?;

    let final_population = draw_population_graphs(descriptor, &run.final_population)?;

    let avg_fitness = run.iterations.iter().map(|i| i.avg_fitness);
    let avg_fitness = rewrap!(minmax_iter(avg_fitness.clone()), |bounds| {
        just_plot_floats("Fitness", bounds, avg_fitness)?
    });

    let best_fitness = run.iterations.iter().map(|i| i.best_fitness);
    let best_fitness = rewrap!(minmax_iter(best_fitness.clone()), |bounds| {
        just_plot_floats("Best fitness", bounds, best_fitness)?
    });

    let fitness_std_dev: Vec<N64> = run.iterations.iter().map(|i| i.fitness_std_dev).collect();
    let fitness_std_dev = rewrap!(minmax_iter(&fitness_std_dev), |(&min, &max)| {
        just_plot_floats("Fitness standard deviation", (min, max), fitness_std_dev)?
    });

    let rr = run.iterations.iter().map(|i| i.rr);
    let theta = run.iterations.iter().map(|i| i.theta);
    let rr_and_theta = rewrap!(minmax_iter(rr.clone().chain(theta.clone())), |bounds| {
        plot_double(bounds, (rr, "RR"), (theta, "Theta"))?
    });

    Ok(RunGraphs {
        starting_population,
        final_population,
        avg_fitness,
        best_fitness,
        fitness_std_dev,
        rr_and_theta,
        optimal_specimen_count: None,
        growth_rate: None,
        selection_diff: None,
        selection_intensity: None,
        selection_intensity_and_diff: None,
    })
}

pub fn draw_run_with_optimum_graphs(
    descriptor: &impl GraphDescriptor,
    run: &RunStatsWithOptimum,
) -> eyre::Result<RunGraphs> {
    let starting_population = run
        .starting_population
        .iter()
        .map(|p| draw_population_graphs(descriptor, p))
        .collect::<Result<_, _>>()?;

    let final_population = draw_population_graphs(descriptor, &run.final_population)?;

    let avg_fitness = run.iterations.iter().map(|i| i.avg_fitness);
    let avg_fitness = rewrap!(minmax_iter(avg_fitness.clone()), |bounds| {
        just_plot_floats("Fitness", bounds, avg_fitness)?
    });

    let best_fitness = run.iterations.iter().map(|i| i.best_fitness);
    let best_fitness = rewrap!(minmax_iter(best_fitness.clone()), |bounds| {
        just_plot_floats("Best fitness", bounds, best_fitness)?
    });

    let fitness_std_dev = run.iterations.iter().map(|i| i.fitness_std_dev);
    let fitness_std_dev = rewrap!(minmax_iter(fitness_std_dev.clone()), |bounds| {
        just_plot_floats("Fitness standard deviation", bounds, fitness_std_dev)?
    });

    let optimal_specimen_count = run.iterations.iter().map(|i| i.optimal_specimen_count);
    let optimal_specimen_count = rewrap!(minmax_iter(optimal_specimen_count.clone()), |bounds| {
        just_plot_ints("Optimal specimen count", bounds, optimal_specimen_count)?
    });

    let growth_rate = run.iterations.iter().map(|i| i.growth_rate);
    let growth_rate = rewrap!(minmax_iter(growth_rate.clone()), |bounds| {
        just_plot_floats("Growth rate", bounds, growth_rate)?
    });

    let rr = run.iterations.iter().map(|i| i.rr);
    let theta = run.iterations.iter().map(|i| i.theta);
    let rr_and_theta = rewrap!(minmax_iter(rr.clone().chain(theta.clone())), |bounds| {
        plot_double(bounds, (rr, "RR"), (theta, "Theta"))?
    });

    let selection_intensity = run.iterations.iter().map(|i| i.selection_intensity);
    let selection_diff = run.iterations.iter().map(|i| i.selection_diff);
    let selection_intensity_and_diff = rewrap!(
        minmax_iter(selection_intensity.clone().chain(selection_diff.clone())),
        |bounds| {
            plot_double(
                bounds,
                (selection_intensity.clone(), "Selection intensity"),
                (selection_diff.clone(), "Selection difference"),
            )?
        }
    );

    let selection_intensity = rewrap!(minmax_iter(selection_intensity.clone()), |bounds| {
        just_plot_floats("Selection intensity", bounds, selection_intensity)?
    });

    let selection_diff = rewrap!(minmax_iter(selection_diff.clone()), |bounds| {
        just_plot_floats("Selection difference", bounds, selection_diff)?
    });

    Ok(RunGraphs {
        starting_population,
        final_population,
        avg_fitness,
        best_fitness,
        selection_intensity,
        selection_diff,
        fitness_std_dev,
        optimal_specimen_count,
        growth_rate,
        rr_and_theta,
        selection_intensity_and_diff,
    })
}

pub fn draw_graphs(
    descriptor: &impl GraphDescriptor,
    stats: &ConfigStats,
) -> eyre::Result<Vec<RunGraphs>> {
    match stats {
        OptimumDisabiguity::WithOptimum(stats) => stats
            .runs
            .iter()
            .take(5)
            .map(|run| draw_run_with_optimum_graphs(descriptor, run))
            .collect(),
        OptimumDisabiguity::Optimumless(stats) => stats
            .runs
            .iter()
            .take(5)
            .map(|run| draw_optimumless_run_graphs(descriptor, run))
            .collect(),
    }
}
