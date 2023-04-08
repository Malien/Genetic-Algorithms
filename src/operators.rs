use rand::{seq::SliceRandom, Rng};

use crate::{EvaluatedGenome, Genome, RunState};

pub fn crossover(state: &mut RunState, mut population: Vec<EvaluatedGenome>) -> Vec<Genome> {
    population.shuffle(&mut state.rng);
    if !state.config.apply_crossover {
        return population.into_iter().map(|genome| genome.genome).collect();
    }
    if population.len() % 2 != 0 {
        panic!("Population size must be even for crossover");
    }
    population
        .into_iter()
        .array_chunks::<2>()
        .flat_map(|[parent1, parent2]| {
            let crossover_point = state.rng.gen_range(0..state.config.genome_len as usize);
            let (child1, child2) = gene_crossover(parent1.genome, parent2.genome, crossover_point);
            [child1, child2]
        })
        .collect()
}

fn gene_crossover(parent1: Genome, parent2: Genome, crossover_point: usize) -> (Genome, Genome) {
    assert!(parent1.len() == parent2.len());
    let len = parent1.len();
    let mut child1 = parent1;
    let mut child2 = parent2;
    child1[crossover_point..len].swap_with_bitslice(&mut child2[crossover_point..len]);
    (child1, child2)
}

#[test]
fn test_gene_crossover() {
    use bitvec::{bitarr, order::Lsb0};

    let parent1 = Genome::Ten(bitarr![u16, Lsb0; 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    let parent2 = Genome::Ten(bitarr![u16, Lsb0; 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let (child1, child2) = gene_crossover(parent1, parent2, 4);
    assert_eq!(&child1[..], &bitarr![u16, Lsb0; 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&child2[..], &bitarr![u16, Lsb0; 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
}

pub fn mutation(state: &mut RunState, population: Vec<Genome>) -> Vec<Genome> {
    if let Some(mutation_rate) = state.config.mutation_rate {
        population
            .into_iter()
            .map(|genome| gene_mutation(genome, mutation_rate, &mut state.rng))
            .collect()
    } else {
        population
    }
}

fn gene_mutation(mut genome: Genome, mutation_rate: f64, rng: &mut impl rand::Rng) -> Genome {
    for i in 0..10 {
        if rng.gen_bool(mutation_rate) {
            let gene = genome[i];
            genome.set(i, !gene);
        }
    }
    genome
}
