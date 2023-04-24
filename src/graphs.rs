use std::path::Path;

use decorum::N64;
use futures::try_join;
use num_traits::real::Real;
use once_cell::sync::Lazy;
use plotters::{
    backend::{BitMapBackend, PixelFormat, RGBPixel},
    chart::ChartBuilder,
    coord::{combinators::{IntoLinspace, IntoLogRange}, Shift},
    drawing::IntoDrawingArea,
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
    graph_image(|root| {
        let mut chart = ChartBuilder::on(&root)
            .caption("Amount of ones in genome", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(20)
            .build_cartesian_2d(0..one_counts.iter().max().unwrap() + 1, 0..one_counts.len())?;

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
                (bounds.min_f()..bounds.max_f())
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

fn just_plot_floats(
    name: &str,
    (min, max): (N64, N64),
    iter: impl ExactSizeIterator<Item = N64> + Clone,
) -> eyre::Result<Vec<u8>> {
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

fn just_plot_ints(
    name: &str,
    (min, max): (usize, usize),
    iter: impl ExactSizeIterator<Item = usize> + Clone,
) -> eyre::Result<Vec<u8>> {
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
            .caption(format!("{a_name} and {b_name}"), ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0..(a.len() - 1), f64::from(min)..f64::from(max))?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(a.map(f64::from).enumerate(), &colors::RED))?.label(a_name);
        chart.draw_series(LineSeries::new(b.map(f64::from).enumerate(), &colors::BLUE))?.label(b_name);

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
    buf: Option<eyre::Result<Vec<u8>>>,
    filename: &str,
) -> eyre::Result<()> {
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

    let mut ones_count = None;
    let mut phenotype = None;
    let mut fitness = None;
    THREAD_POOL.scope(|s| {
        s.spawn(|_| {
            ones_count = Some(population_ones_count_graph(&population.ones_count));
        });
        s.spawn(|_| {
            let Some(bounds) = phenotype_bounds else { return };
            let Some(phenotypes) = population.phenotype.as_ref() else { return };
            phenotype = Some(population_phenotype_graph(bounds, phenotypes));
        });
        s.spawn(|_| {
            let Some(bounds) = fitness_bounds else { return };
            fitness = Some(population_fitness_graph(bounds, &population.fitness));
        });
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

    let mut avg_fitness = None;
    let mut best_fitness = None;
    let mut fitness_std_dev = None;
    let mut rr_and_theta = None;

    THREAD_POOL.scope(|s| {
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.avg_fitness);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            avg_fitness = Some(just_plot_floats("Fitness", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.best_fitness);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            best_fitness = Some(just_plot_floats("Best fitness", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.fitness_std_dev);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            fitness_std_dev = Some(just_plot_floats("Fitness std dev", bounds, iter));
        });
        s.spawn(|_| {
            let rr = run.iterations.iter().map(|i| i.rr);
            let theta = run.iterations.iter().map(|i| i.theta);
            let Some(bounds) = minmax_iter(rr.clone().chain(theta.clone())) else { return };
            rr_and_theta = Some(plot_double(bounds, (rr, "RR"), (theta, "Theta")));
        });
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

    let mut avg_fitness = None;
    let mut best_fitness = None;
    let mut selection_intensity = None;
    let mut selection_diff = None;
    let mut fitness_std_dev = None;
    let mut optimal_specimen_count = None;
    let mut growth_rate = None;
    let mut rr_and_theta = None;
    let mut selection_intensity_and_diff = None;

    THREAD_POOL.scope(|s| {
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.avg_fitness);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            avg_fitness = Some(just_plot_floats("Fitness", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.best_fitness);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            best_fitness = Some(just_plot_floats("Best fitness", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.selection_intensity);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            selection_intensity = Some(just_plot_floats("Selection intensity", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.selection_diff);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            selection_diff = Some(just_plot_floats("Selection diff", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.fitness_std_dev);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            fitness_std_dev = Some(just_plot_floats("Fitness std dev", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.optimal_specimen_count);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            optimal_specimen_count = Some(just_plot_ints("Optimal specimen count", bounds, iter));
        });
        s.spawn(|_| {
            let iter = run.iterations.iter().map(|i| i.growth_rate);
            let Some(bounds) = minmax_iter(iter.clone()) else { return };
            growth_rate = Some(just_plot_floats("Growth rate", bounds, iter));
        });
        s.spawn(|_| {
            let rr = run.iterations.iter().map(|i| i.rr);
            let theta = run.iterations.iter().map(|i| i.theta);
            let Some(bounds) = minmax_iter(rr.clone().chain(theta.clone())) else { return };
            rr_and_theta = Some(plot_double(bounds, (rr, "RR"), (theta, "Theta")));
        });
        s.spawn(|_| {
            let intensity = run.iterations.iter().map(|i| i.selection_intensity);
            let diff = run.iterations.iter().map(|i| i.selection_diff);
            let Some(bounds) = minmax_iter(intensity.clone().chain(diff.clone())) else { return };
            selection_intensity_and_diff = Some(plot_double(bounds, (intensity, "Selection intensity"), (diff, "Selection difference")));
        });
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
