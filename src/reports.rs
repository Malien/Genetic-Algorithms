use std::{
    collections::{hash_map::Entry, HashMap},
    sync::atomic::Ordering,
    thread::ThreadId,
    time::Duration,
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::{persistance::ConfigKey, AlgoDescriptor, MaterializedDescriptor, SILENT, VERBOSE};

pub struct Report {
    kind: ReportKind,
    key: ConfigKey<MaterializedDescriptor>,
}

enum ReportKind {
    Solving(ThreadId),
    TakingTooLong {
        thread: ThreadId,
        completion: usize,
        out_of: usize,
    },
    Graphing(ThreadId),
    End(ThreadId),
    CuttingShort {
        failed_runs: usize,
    },
    WroteTables,
    WroteGraphs,
}

impl Report {
    pub fn solving(thread: ThreadId, key: ConfigKey<impl AlgoDescriptor>) -> Self {
        Self {
            kind: ReportKind::Solving(thread),
            key: ConfigKey {
                algo_type: key.algo_type.materialize(),
                ..key
            },
        }
    }

    pub fn taking_too_long(
        thread: ThreadId,
        key: ConfigKey<impl AlgoDescriptor>,
        (completion, out_of): (usize, usize),
    ) -> Self {
        Self {
            kind: ReportKind::TakingTooLong {
                thread,
                completion,
                out_of,
            },
            key: ConfigKey {
                algo_type: key.algo_type.materialize(),
                ..key
            },
        }
    }

    pub fn graphing(thread: ThreadId, key: ConfigKey<impl AlgoDescriptor>) -> Self {
        Self {
            kind: ReportKind::Graphing(thread),
            key: ConfigKey {
                algo_type: key.algo_type.materialize(),
                ..key
            },
        }
    }

    pub fn end(thread: ThreadId, key: ConfigKey<impl AlgoDescriptor>) -> Self {
        Self {
            kind: ReportKind::End(thread),
            key: ConfigKey {
                algo_type: key.algo_type.materialize(),
                ..key
            },
        }
    }

    pub fn cutting_short(key: ConfigKey<impl AlgoDescriptor>, failed_runs: usize) -> Self {
        Self {
            kind: ReportKind::CuttingShort { failed_runs },
            key: ConfigKey {
                algo_type: key.algo_type.materialize(),
                ..key
            },
        }
    }

    pub fn wrote_tables(key: ConfigKey<impl AlgoDescriptor>) -> Self {
        Self {
            kind: ReportKind::WroteTables,
            key: ConfigKey {
                algo_type: key.algo_type.materialize(),
                ..key
            },
        }
    }

    pub fn wrote_graphs(key: ConfigKey<impl AlgoDescriptor>) -> Self {
        Self {
            kind: ReportKind::WroteGraphs,
            key: ConfigKey {
                algo_type: key.algo_type.materialize(),
                ..key
            },
        }
    }
}

enum ThreadStatus {
    Solving,
    SolvingTooLong(usize, usize),
    Graphing,
    Stalled,
}

struct ThreadState {
    status: ThreadStatus,
    key: ConfigKey<MaterializedDescriptor>,
    bar: ProgressBar,
}

static THREAD_BAR_STYLE: Lazy<ProgressStyle> = Lazy::new(|| {
    ProgressStyle::default_spinner()
        .template("{prefix:12}: {spinner} {msg:<100}")
        .unwrap()
});

static THREAD_STALLED_BAR_STYLE: Lazy<ProgressStyle> = Lazy::new(|| {
    ProgressStyle::default_spinner()
        .template("{prefix:12}: {spinner} {msg:<100} ⏳[{bar:5.red/black}]")
        .unwrap()
});

pub fn report(report: Report, tx: &UnboundedSender<Report>) {
    if SILENT.load(Ordering::Relaxed) {
        return;
    }
    let _ = tx.send(report);
}

pub async fn run_reports(config_len: u64, mut rx: UnboundedReceiver<Report>) -> eyre::Result<()> {
    if SILENT.load(Ordering::Relaxed) {
        return Ok(());
    }

    let bar_style =
        ProgressStyle::with_template("{prefix:12}: [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")?;

    let multibar = MultiProgress::new();
    let solve_bar = multibar.add(
        ProgressBar::new(config_len)
            .with_prefix("Solved")
            .with_style(bar_style.clone()),
    );
    let graph_bar = multibar.add(
        ProgressBar::new(config_len)
            .with_prefix("Graphed")
            .with_style(bar_style.clone()),
    );
    let write_table_bar = multibar.add(
        ProgressBar::new(config_len)
            .with_prefix("Table writes")
            .with_style(bar_style.clone()),
    );
    let write_graph_bar = multibar.add(
        ProgressBar::new(config_len)
            .with_prefix("Graph writes")
            .with_style(bar_style.clone()),
    );

    let mut solved = 0;
    let mut graphed = 0;
    let mut wrote_tables = 0;
    let mut wrote_graphs = 0;
    let verbose = VERBOSE.load(Ordering::Relaxed);

    let mut thread_statuses = HashMap::<ThreadId, ThreadState>::new();

    let mut update_thread =
        |thread: ThreadId, key: ConfigKey<MaterializedDescriptor>, status: ThreadStatus| {
            match thread_statuses.entry(thread) {
                Entry::Occupied(mut entry) => {
                    let state = entry.get_mut();
                    state.status = status;
                    state.key = key;
                    update_thread_message(state);
                }
                Entry::Vacant(entry) => {
                    let bar = ProgressBar::new_spinner().with_prefix(format!("{thread:?}"));
                    bar.enable_steady_tick(Duration::from_secs(1));
                    let mut state = ThreadState {
                        status,
                        key,
                        bar: multibar.insert_before(&solve_bar, bar),
                    };
                    update_thread_message(&mut state);
                    entry.insert(state);
                }
            }
        };

    while let Some(Report { kind, key }) = rx.recv().await {
        use ReportKind::*;
        match kind {
            Solving(thread) => {
                update_thread(thread, key, ThreadStatus::Solving);
            }
            Graphing(thread) => {
                solved += 1;
                update_thread(thread, key, ThreadStatus::Graphing);
                solve_bar.set_position(solved);
                if verbose {
                    multibar.println(format!("Solved: {key}"))?;
                }
            }
            TakingTooLong {
                thread,
                completion,
                out_of,
            } => {
                update_thread(
                    thread,
                    key,
                    ThreadStatus::SolvingTooLong(completion, out_of),
                );
                if verbose {
                    multibar.println(format!(
                        "Solving {key} is taking too long ({completion}/{out_of})"
                    ))?;
                }
            }
            End(thread) => {
                graphed += 1;
                update_thread(thread, key, ThreadStatus::Stalled);
                graph_bar.set_position(graphed);
                if verbose {
                    multibar.println(format!("Graphed: {key}"))?;
                }
            }
            CuttingShort { failed_runs } if verbose => {
                multibar.println(format!("First {failed_runs} didn't converge. Skipping {key}"))?;
            }
            CuttingShort { .. } => {}
            WroteTables => {
                wrote_tables += 1;
                write_table_bar.set_position(wrote_tables);
                if verbose {
                    multibar.println(format!("Wrote tables: {key}"))?;
                }
            }
            WroteGraphs => {
                wrote_graphs += 1;
                write_graph_bar.set_position(wrote_graphs);
                if verbose {
                    multibar.println(format!("Wrote graphs: {key}"))?;
                }
            }
        }
    }

    Ok(())
}

fn update_thread_message(state: &mut ThreadState) {
    match state.status {
        ThreadStatus::Solving => {
            state.bar.set_style(THREAD_BAR_STYLE.clone());
            state.bar.set_message(format!("Solving:  {}", state.key));
        }
        ThreadStatus::SolvingTooLong(progress, out_of) => {
            state.bar.set_style(THREAD_STALLED_BAR_STYLE.clone());
            state.bar.set_length(out_of as u64);
            state.bar.set_position(progress as u64);
        }
        ThreadStatus::Graphing => {
            state.bar.set_style(THREAD_BAR_STYLE.clone());
            state.bar.set_message(format!("Graphing: {}", state.key));
        }
        ThreadStatus::Stalled => {
            state.bar.set_style(THREAD_BAR_STYLE.clone());
            state.bar.set_message(format!("Stalled"));
        }
    }
}
