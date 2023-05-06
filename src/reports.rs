use std::{
    collections::{hash_map::Entry, HashMap},
    thread::ThreadId,
    time::Duration,
    sync::atomic::Ordering,
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::{persistance::ConfigKey, AlgoDescriptor, MaterializedDescriptor, VERBOSE, SILENT};

pub struct Report {
    kind: ReportKind,
    key: ConfigKey<MaterializedDescriptor>,
}

enum ReportKind {
    Solving(ThreadId),
    Graphing(ThreadId),
    End(ThreadId),
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
    Graphing,
    Stalled,
}

struct ThreadState {
    status: ThreadStatus,
    bar: ProgressBar,
}

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
    let spinner_style = ProgressStyle::with_template("{prefix:12}: {spinner} {msg}")?;

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
            .with_style(bar_style),
    );

    let mut solved = 0;
    let mut graphed = 0;
    let mut wrote_tables = 0;
    let mut wrote_graphs = 0;
    let verbose = VERBOSE.load(Ordering::Relaxed);

    let mut thread_statuses = HashMap::<ThreadId, ThreadState>::new();

    let mut update_thread =
        |thread: ThreadId, status: ThreadStatus, msg: String| match thread_statuses.entry(thread) {
            Entry::Occupied(mut entry) => {
                let state = entry.get_mut();
                state.status = status;
                state.bar.set_message(msg);
            }
            Entry::Vacant(entry) => {
                let bar = ProgressBar::new_spinner()
                    .with_style(spinner_style.clone())
                    .with_prefix(format!("{thread:?}"))
                    .with_message(msg);
                bar.enable_steady_tick(Duration::from_secs(1));
                entry.insert(ThreadState {
                    status,
                    bar: multibar.insert_before(&solve_bar, bar),
                });
            }
        };

    while let Some(Report { kind, key }) = rx.recv().await {
        use ReportKind::*;
        match kind {
            Solving(thread) => {
                update_thread(thread, ThreadStatus::Solving, format!("Solving:  {key}"));
            }
            Graphing(thread) => {
                solved += 1;
                update_thread(thread, ThreadStatus::Graphing, format!("Graphing: {key}"));
                solve_bar.set_position(solved);
                if verbose {
                    multibar.println(format!("Solved: {key}"))?;
                }
            }
            End(thread) => {
                graphed += 1;
                update_thread(thread, ThreadStatus::Stalled, format!("Stalled"));
                graph_bar.set_position(graphed);
                if verbose {
                    multibar.println(format!("Graphed: {key}"))?;
                }
            }
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
