use std::collections::HashSet;
use std::fmt;

use decorum::{Finite, R64};
use rand::seq::{SliceRandom, index::sample};

use crate::{RunState, EvaluatedGenome, FullFamily};

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

pub fn selection<F: FullFamily>(state: &mut RunState<F>, population: Vec<EvaluatedGenome<{F::N}>>) -> SelectionResult<{F::N}> 
where [(); bitvec::mem::elts::<u16>(F::N)]:
{
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

pub struct SelectionResult<const N: usize> 
where [(); bitvec::mem::elts::<u16>(N)]:
{
    pub new_population: Vec<EvaluatedGenome<N>>,
    pub unique_specimens_selected: usize,
}

fn stochastic_tournament_selection_without_replacement<const N: usize>(
    initial_population: Vec<EvaluatedGenome<N>>,
    chance: R64,
    rng: &mut impl rand::Rng,
) -> SelectionResult<N> 
where [(); bitvec::mem::elts::<u16>(N)]:
{
    let len = initial_population.len();
    assert!(len % 2 == 0, "Population size must be even. Given vec of size {}", initial_population.len());
    let mut population = Vec::with_capacity(len);
    population.extend(initial_population.into_iter().enumerate());

    let mut new_population = Vec::with_capacity(len);
    let mut winner_indecies = HashSet::with_capacity(len);

    population.shuffle(rng);
    for [&specimen_a, &specimen_b] in population.iter().array_chunks::<2>() {
        let (looser, winner) = minmax_by(specimen_a, specimen_b, |(_idx, specimen)| specimen.fitness);
        let ((winner_idx, winner), _) = reroll_winner(winner, looser, chance, rng);
        new_population.push(winner);
        winner_indecies.insert(winner_idx);
    }

    population.shuffle(rng);
    for [specimen_a, specimen_b] in population.into_iter().array_chunks::<2>() {
        let (looser, winner) = minmax_by(specimen_a, specimen_b, |(_idx, specimen)| specimen.fitness);
        let ((winner_idx, winner), _) = reroll_winner(winner, looser, chance, rng);
        new_population.push(winner);
        winner_indecies.insert(winner_idx);
    }

    SelectionResult {
        new_population,
        unique_specimens_selected: winner_indecies.len(),
    }
}

fn stochastic_tournament_selection_with_replacement<const N: usize>(
    initial_population: Vec<EvaluatedGenome<N>>,
    chance: R64,
    rng: &mut impl rand::Rng,
) -> SelectionResult<N> 
where [(); bitvec::mem::elts::<u16>(N)]:
{
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
