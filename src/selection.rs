use std::collections::HashSet;
use std::fmt;

use decorum::{Finite, R64};
use rand::seq::{SliceRandom, index::sample};

use crate::{RunState, EvaluatedGenome};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Selection {
    StochasticTournament {
        prob: Finite<f64>,
        replacement: TournamentReplacement,
    },
}

impl fmt::Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Selection::StochasticTournament { prob, replacement } => {
                write!(f, "stochastic-tournament-{}-{:.1}", replacement, prob)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TournamentReplacement {
    With,
    Without,
}

impl fmt::Display for TournamentReplacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TournamentReplacement::With => write!(f, "with-replacement"),
            TournamentReplacement::Without => write!(f, "without-replacement"),
        }
    }
}

pub fn selection(state: &mut RunState, population: Vec<EvaluatedGenome>) -> SelectionResult {
    let Selection::StochasticTournament { prob, replacement } = state.config.selection;
    match replacement {
        TournamentReplacement::With => {
            stochastic_tournament_selection_with_replacement(population, prob, &mut state.rng)
        }
        TournamentReplacement::Without => {
            stochastic_tournament_selection_without_replacement(population, prob, &mut state.rng)
        }
    }
}

pub struct SelectionResult {
    pub new_population: Vec<EvaluatedGenome>,
    pub unique_specimens_selected: usize,
}

fn stochastic_tournament_selection_with_replacement(
    initial_population: Vec<EvaluatedGenome>,
    chance: R64,
    rng: &mut impl rand::Rng,
) -> SelectionResult {
    let mut population = Vec::with_capacity(initial_population.len() * 2);
    population.extend(initial_population.iter().cloned().enumerate());
    population.extend(initial_population.into_iter().enumerate());
    population.shuffle(rng);
    let winners = population
        .into_iter()
        .array_chunks::<2>()
        .map(|[specimen_a, specimen_b]| {
            let (looser, winner) = minmax_by(specimen_a, specimen_b, |(_idx, specimen)| specimen.fitness);
            let (winner, _) = reroll_winner(winner, looser, chance, rng);
            winner
        });

    let mut new_population = Vec::with_capacity(winners.len());
    let mut winner_indecies = HashSet::with_capacity(winners.len());
    for (winner_idx, winner) in winners {
        new_population.push(winner);
        winner_indecies.insert(winner_idx);
    }

    SelectionResult {
        new_population,
        unique_specimens_selected: winner_indecies.len(),
    }
}

fn stochastic_tournament_selection_without_replacement(
    initial_population: Vec<EvaluatedGenome>,
    chance: R64,
    rng: &mut impl rand::Rng,
) -> SelectionResult {
    let mut new_population = Vec::with_capacity(initial_population.len());
    let mut winner_indecies = HashSet::with_capacity(initial_population.len());

    for _ in 0..initial_population.len() {
        let indecies = sample(rng, initial_population.len(), 2);
        let idx_a = indecies.index(0);
        let idx_b = indecies.index(1);
        let specimen_a = (idx_a, &initial_population[idx_a]);
        let specimen_b = (idx_b, &initial_population[idx_b]);
        let (looser, winner) = minmax_by(specimen_a, specimen_b, |(_idx, specimen)| specimen.fitness);
        let ((winner_idx, winner), _) = reroll_winner(winner, looser, chance, rng);
        new_population.push(winner.clone());
        winner_indecies.insert(winner_idx);
    }

    SelectionResult {
        new_population,
        unique_specimens_selected: winner_indecies.len(),
    }
}

fn minmax_by<T, U: PartialOrd>(lhs: T, rhs: T, key_fn: impl Fn(&T) -> U) -> (T, T) {
    if key_fn(&lhs) < key_fn(&rhs) {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}


fn reroll_winner<T>(winner: T, looser: T, chance: R64, rng: &mut impl rand::Rng) -> (T, T) {
    if rng.gen_bool(chance.into()) {
        (winner, looser)
    } else {
        (looser, winner)
    }
}
