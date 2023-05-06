use rand::{seq::SliceRandom, Rng};

use crate::{EvaluatedGenome, Genome, RunState, FullFamily, GenFamily};

pub fn crossover<F: GenFamily>(state: &mut RunState<F>, mut population: Vec<EvaluatedGenome<{F::N}>>) -> Vec<Genome<{F::N}>> 
where [(); bitvec::mem::elts::<u16>(F::N)]:
{
    if !state.config.apply_crossover {
        return population.into_iter().map(|genome| genome.genome).collect();
    }
    if population.len() % 2 != 0 {
        panic!("Population size must be even for crossover");
    }
    population.shuffle(&mut state.rng);
    population
        .into_iter()
        .array_chunks::<2>()
        .flat_map(|[parent1, parent2]| {
            let crossover_point = state.rng.gen_range(0..F::N);
            let (child1, child2) = gene_crossover(parent1.genome, parent2.genome, crossover_point);
            [child1, child2]
        })
        .collect()
}

fn gene_crossover<const N: usize>(parent1: Genome<N>, parent2: Genome<N>, crossover_point: usize) -> (Genome<N>, Genome<N>) 
where [(); bitvec::mem::elts::<u16>(N)]:
{
    let len = parent1.len();
    let mut child1 = parent1;
    let mut child2 = parent2;
    child1[crossover_point..len].swap_with_bitslice(&mut child2[crossover_point..len]);
    (child1, child2)
}

#[test]
fn test_gene_crossover() {
    use bitvec::{bitarr, order::Lsb0};

    let parent1 = Genome::new(bitarr![u16, Lsb0; 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    let parent2 = Genome::new(bitarr![u16, Lsb0; 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let (child1, child2) = gene_crossover::<10>(parent1, parent2, 4);
    assert_eq!(&child1[..], &bitarr![u16, Lsb0; 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&child2[..], &bitarr![u16, Lsb0; 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
}

pub fn mutation<F: FullFamily>(state: &mut RunState<F>, mut population: Vec<Genome<{F::N}>>) -> Vec<Genome<{F::N}>> 
where [(); bitvec::mem::elts::<u16>(F::N)]:
{
    let Some(mutation_rate) = state.config.mutation_rate else { return population; };

    for genome in &mut population {
        for i in 0..F::N {
            if state.rng.gen_bool(mutation_rate) {
                let gene = genome[i];
                genome.set(i, !gene);
            }
        }
    }
    return population;
}
