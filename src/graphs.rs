use std::path::Path;

use decorum::N64;
use futures::try_join;
use num_traits::real::Real;
use once_cell::sync::Lazy;
use plotters::{
    backend::{BitMapBackend, PixelFormat, RGBPixel},
    chart::ChartBuilder,
    coord::{
        combinators::{IntoLinspace, IntoLogRange},
        Shift,
    },
    drawing::IntoDrawingArea,
    prelude::PathElement,
    series::{Histogram, LineSeries},
    style::{colors, Color, IntoFont},
};
use tokio::io::AsyncWriteExt;

use crate::stats::{OptimumlessRunStats, PopulationStats, RunStatsWithOptimum};
use crate::{BinaryAlgo, GenomeEncoding, PheonotypeAlgo};

const GRAPH_WIDTH: u32 = 600;
const GRAPH_HEIGHT: u32 = 400;

type DrawingArea<'a> = plotters::drawing::DrawingArea<BitMapBackend<'a, RGBPixel>, Shift>;

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
        CompressionType::Best,
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

async fn write_file(
    file_limiter: &tokio::sync::Semaphore,
    basename: &Path,
    buf: tokio::task::JoinHandle<Option<eyre::Result<Vec<u8>>>>,
    filename: &str,
) -> eyre::Result<()> {
    let buf = buf.await?;
    let _permit = file_limiter.acquire().await;
    let Some(buf) = buf else { return Ok(()) };
    let buf = buf?;
    let mut file = tokio::fs::File::create(basename.join(filename)).await?;
    file.write_all(&buf).await?;
    file.flush().await?;
    // Let's try this to let tokio properly close file handles
    tokio::task::yield_now().await;
    Ok(())
}

macro_rules! write_all {
    ($file_limiter:expr, $path:expr; $($future:ident),+ $(,)?) => {
        try_join!(
            $(write_file($file_limiter, $path, $future, concat!(stringify!($future), ".png"))),+
        )
    };
}

// A separate thread pool for drawing graphs
static THREAD_POOL: Lazy<rayon::ThreadPool> =
    Lazy::new(|| rayon::ThreadPoolBuilder::new().build().unwrap());

pub async fn draw_population_graphs(
    file_limiter: &tokio::sync::Semaphore,
    descriptor: impl GraphDescriptor,
    path: &Path,
    population: &PopulationStats,
) -> eyre::Result<()> {
    tokio::fs::create_dir_all(&path).await?;

    let phenotype_bounds = descriptor.phenotype_bounds();
    let fitness_bounds = descriptor.fitness_bounds();

    let ones_count = population.ones_count.to_vec();
    let ones_count =
        tokio::task::spawn_blocking(move || Some(population_ones_count_graph(&ones_count)));

    let phenotype = population.phenotype.clone();
    let phenotype = tokio::task::spawn_blocking(move || {
        let bounds = phenotype_bounds?;
        Some(population_phenotype_graph(bounds, &phenotype?))
    });

    let fitness = population.fitness.to_vec();
    let fitness = tokio::task::spawn_blocking(move || {
        let bounds = fitness_bounds?;
        Some(population_fitness_graph(bounds, &fitness))
    });

    write_all!(file_limiter, &path; ones_count, phenotype, fitness)?;

    Ok(())
}

pub async fn draw_optimumless_run_graphs(
    file_limiter: &tokio::sync::Semaphore,
    descriptor: impl GraphDescriptor + Clone,
    path: &Path,
    run: &OptimumlessRunStats,
) -> eyre::Result<()> {
    tokio::fs::create_dir_all(&path).await?;

    for (i, population) in run.starting_population.iter().enumerate() {
        draw_population_graphs(
            file_limiter,
            descriptor.clone(),
            &path.join(i.to_string()),
            population,
        )
        .await?;
    }

    draw_population_graphs(
        file_limiter,
        descriptor,
        &path.join("final"),
        &run.final_population,
    )
    .await?;

    let avg_fitness: Vec<N64> = run.iterations.iter().map(|i| i.avg_fitness).collect();
    let avg_fitness = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&avg_fitness)?;
        Some(just_plot_floats("Fitness", (min, max), avg_fitness))
    });

    let best_fitness: Vec<N64> = run.iterations.iter().map(|i| i.best_fitness).collect();
    let best_fitness = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&best_fitness)?;
        Some(just_plot_floats("Best fitness", (min, max), best_fitness))
    });

    let fitness_std_dev: Vec<N64> = run.iterations.iter().map(|i| i.fitness_std_dev).collect();
    let fitness_std_dev = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(fitness_std_dev.iter())?;
        Some(just_plot_floats(
            "Fitness standard deviation",
            (min, max),
            fitness_std_dev,
        ))
    });

    let rr: Vec<N64> = run.iterations.iter().map(|i| i.rr).collect();
    let theta: Vec<N64> = run.iterations.iter().map(|i| i.theta).collect();
    let rr_and_theta = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(rr.iter().chain(&theta))?;
        Some(plot_double(
            (min, max),
            (rr.into_iter(), "RR"),
            (theta.into_iter(), "Theta"),
        ))
    });

    write_all!(file_limiter, &path; avg_fitness, best_fitness, fitness_std_dev, rr_and_theta)?;

    Ok(())
}

pub async fn draw_run_with_optimum_graphs(
    file_limiter: &tokio::sync::Semaphore,
    descriptor: impl GraphDescriptor + Clone,
    path: &Path,
    run: &RunStatsWithOptimum,
) -> eyre::Result<()> {
    tokio::fs::create_dir_all(&path).await?;

    for (i, population) in run.starting_population.iter().enumerate() {
        draw_population_graphs(
            file_limiter,
            descriptor.clone(),
            &path.join(i.to_string()),
            population,
        )
        .await?;
    }

    draw_population_graphs(
        file_limiter,
        descriptor,
        &path.join("final"),
        &run.final_population,
    )
    .await?;

    let avg_fitness: Vec<_> = run.iterations.iter().map(|i| i.avg_fitness).collect();
    let avg_fitness = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&avg_fitness)?;
        Some(just_plot_floats("Fitness", (min, max), avg_fitness))
    });

    let best_fitness: Vec<_> = run.iterations.iter().map(|i| i.best_fitness).collect();
    let best_fitness = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&best_fitness)?;
        Some(just_plot_floats("Best fitness", (min, max), best_fitness))
    });

    let selection_diff: Vec<_> = run.iterations.iter().map(|i| i.selection_diff).collect();
    let selection_diff = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&selection_diff)?;
        Some(just_plot_floats(
            "Selection difference",
            (min, max),
            selection_diff,
        ))
    });

    let fitness_std_dev: Vec<_> = run.iterations.iter().map(|i| i.fitness_std_dev).collect();
    let fitness_std_dev = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&fitness_std_dev)?;
        Some(just_plot_floats(
            "Fitness standard deviation",
            (min, max),
            fitness_std_dev,
        ))
    });

    let optimal_specimen_count: Vec<_> = run
        .iterations
        .iter()
        .map(|i| i.optimal_specimen_count)
        .collect();
    let optimal_specimen_count = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&optimal_specimen_count)?;
        Some(just_plot_ints(
            "Optimal specimen count",
            (min, max),
            optimal_specimen_count,
        ))
    });

    let growth_rate: Vec<_> = run.iterations.iter().map(|i| i.growth_rate).collect();
    let growth_rate = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&growth_rate)?;
        Some(just_plot_floats("Growth rate", (min, max), growth_rate))
    });

    let rr: Vec<_> = run.iterations.iter().map(|i| i.rr).collect();
    let theta: Vec<_> = run.iterations.iter().map(|i| i.theta).collect();
    let rr_and_theta = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(rr.iter().chain(&theta))?;
        Some(plot_double(
            (min, max),
            (rr.into_iter(), "RR"),
            (theta.into_iter(), "Theta"),
        ))
    });

    let selection_intensity: Vec<_> = run
        .iterations
        .iter()
        .map(|i| i.selection_intensity)
        .collect();
    let selection_diff: Vec<_> = run.iterations.iter().map(|i| i.selection_diff).collect();
    let selection_intensity_and_diff = tokio::task::spawn_blocking({
        let selection_intensity = selection_intensity.clone();
        let selection_diff = selection_diff.clone();
        move || {
            let (&min, &max) = minmax_iter(selection_intensity.iter().chain(&selection_diff))?;
            Some(plot_double(
                (min, max),
                (selection_intensity.into_iter(), "Selection intensity"),
                (selection_diff.into_iter(), "Selection difference"),
            ))
        }
    });

    let selection_intensity = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&selection_intensity)?;
        Some(just_plot_floats(
            "Selection intensity",
            (min, max),
            selection_intensity,
        ))
    });

    let selection_diff = tokio::task::spawn_blocking(move || {
        let (&min, &max) = minmax_iter(&selection_diff)?;
        Some(just_plot_floats(
            "Selection difference",
            (min, max),
            selection_diff,
        ))
    });

    write_all!(file_limiter, &path;
        avg_fitness,
        best_fitness,
        selection_intensity,
        selection_diff,
        fitness_std_dev,
        optimal_specimen_count,
        growth_rate,
        rr_and_theta,
        selection_intensity_and_diff,
    )?;

    Ok(())
}
