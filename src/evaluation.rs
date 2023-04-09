use core::fmt;

use decorum::N64;

use crate::{EvaluatedGenome, GenFamily, Genome, RunState, G10, G100};

pub fn evaluate<F: EvaluateFamily>(
    state: &mut RunState<F>,
    population: Vec<Genome<{ F::N }>>,
) -> Vec<EvaluatedGenome<{ F::N }>>
where
    [(); bitvec::mem::elts::<u16>(F::N)]:,
{
    fn res<const N: usize>(
        population: Vec<Genome<N>>,
        fitness_fn: impl Fn(Genome<N>) -> N64,
    ) -> Vec<EvaluatedGenome<N>>
    where
        [(); bitvec::mem::elts::<u16>(N)]:,
    {
        population
            .into_iter()
            .map(|genome| EvaluatedGenome {
                fitness: fitness_fn(genome),
                genome,
            })
            .collect()
    }

    res(population, |genome| F::evalute(genome, state.config.ty))
}

pub trait EvaluateFamily: GenFamily {
    fn evalute(genome: Genome<{ Self::N }>, algo: Self::AlgoType) -> N64
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:;

    fn decode_phenotype(genome: Genome<{ Self::N }>, algo: Self::AlgoType) -> Option<N64>
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:;
}

impl EvaluateFamily for G10 {
    fn evalute(genome: Genome<10>, (algo, encoding): Self::AlgoType) -> N64
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        let (decode_genome, decode_phenotype, fitness) = phenotype_fns(algo, encoding);
        fitness(decode_phenotype(decode_genome(genome)))
    }

    fn decode_phenotype(genome: Genome<{ Self::N }>, (algo, encoding): Self::AlgoType) -> Option<N64>
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        let (decode_genome, decode_phenotype, _) = phenotype_fns(algo, encoding);
        Some(decode_phenotype(decode_genome(genome)))
    }
}

impl EvaluateFamily for G100 {
    fn evalute(genome: Genome<{ Self::N }>, algo: Self::AlgoType) -> N64
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        match algo {
            BinaryAlgo::FConst => 100.0.into(),
            BinaryAlgo::FHD { sigma } => fhd::fitness(sigma, genome),
        }
    }

    fn decode_phenotype(_: Genome<{ Self::N }>, _: Self::AlgoType) -> Option<N64>
    where
        [(); bitvec::mem::elts::<u16>(Self::N)]:,
    {
        None
    }
}

fn phenotype_fns(
    algo: PheonotypeAlgo,
    encoding: GenomeEncoding,
) -> (
    fn(Genome<10>) -> Genome<10>,
    fn(Genome<10>) -> N64,
    fn(N64) -> N64,
) {
    let decode_phenotype_fn = match algo {
        PheonotypeAlgo::Pow1 => pow1::decode_binary,
        PheonotypeAlgo::Pow2 => pow2::decode_binary,
    };
    let fitness_fn = match algo {
        PheonotypeAlgo::Pow1 => pow1::fitness,
        PheonotypeAlgo::Pow2 => pow2::fitness,
    };
    let decode_genome_fn = match encoding {
        GenomeEncoding::Binary => id,
        GenomeEncoding::BinaryGray => gray_to_binary,
    };
    (decode_genome_fn, decode_phenotype_fn, fitness_fn)
}

fn id<T>(x: T) -> T {
    x
}

pub fn gray_to_binary<const N: usize>(mut genome: Genome<N>) -> Genome<N>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    for i in 1..genome.len() {
        let value = genome[i - 1] ^ genome[i];
        genome.set(i, value);
    }
    genome
}

pub fn binary_to_gray<const N: usize>(mut genome: Genome<N>) -> Genome<N>
where
    [(); bitvec::mem::elts::<u16>(N)]:,
{
    for i in (1..genome.len()).rev() {
        let value = genome[i - 1] ^ genome[i];
        genome.set(i, value);
    }
    genome
}

// #[test]
// fn gray_conversions() {
//     use bitvec::prelude::*;
//     let binary = bitarr![u8, Lsb0; 1, 0, 1];
//     let gray = binary_to_gray(&binary);
//     let binary_again = gray_to_binary(&gray);
//     assert_eq!(gray, bitvec![u8, Lsb0; 1, 1, 1]);
//     assert_eq!(binary, binary_again);
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PheonotypeEvaluation {
    pub fitness: N64,
    pub pheonotype: N64,
}

pub fn evaluate_phenotype(
    algo: PheonotypeAlgo,
    encoding: GenomeEncoding,
    genome: Genome<10>,
) -> PheonotypeEvaluation {
    let (decode_genome, decode_phenotype, fitness) = phenotype_fns(algo, encoding);
    let pheonotype = decode_phenotype(decode_genome(genome));
    PheonotypeEvaluation {
        fitness: fitness(pheonotype),
        pheonotype,
    }
}

pub fn optimal_binary_specimen(algo: BinaryAlgo) -> Option<Genome<100>> {
    match algo {
        BinaryAlgo::FConst => None,
        BinaryAlgo::FHD { .. } => Some(fhd::optimal_specimen()),
    }
}

pub fn optimal_pheonotype_specimen(algo: PheonotypeAlgo) -> Genome<10> {
    match algo {
        PheonotypeAlgo::Pow1 => pow1::optimal_specimen(),
        PheonotypeAlgo::Pow2 => pow2::optimal_specimen(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenomeEncoding {
    Binary,
    BinaryGray,
}

impl fmt::Display for GenomeEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenomeEncoding::Binary => write!(f, "binary"),
            GenomeEncoding::BinaryGray => write!(f, "gray"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryAlgo {
    FConst,
    FHD { sigma: N64 },
}

impl fmt::Display for BinaryAlgo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryAlgo::FConst => write!(f, "FConst"),
            BinaryAlgo::FHD { sigma } => write!(f, "FHD-sigma={sigma}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PheonotypeAlgo {
    Pow1,
    Pow2,
}

impl fmt::Display for PheonotypeAlgo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PheonotypeAlgo::Pow1 => write!(f, "pow1"),
            PheonotypeAlgo::Pow2 => write!(f, "pow2"),
        }
    }
}

pub mod fhd {
    use super::Genome;
    use bitvec::{bitarr, order::Lsb0};
    use decorum::N64;
    use num::FromPrimitive;

    pub fn optimal_specimen() -> Genome<100> {
        Genome::new(bitarr![u16, Lsb0; 0; 100])
    }

    pub fn fitness(sigma: N64, genome: Genome<100>) -> N64 {
        let zeros = N64::from_usize(genome.count_zeros()).unwrap();
        let len = N64::from_usize(genome.len()).unwrap();
        len - zeros + zeros * sigma
    }
}

pub mod pow1 {
    use super::Genome;
    use bitvec::{bitarr, order::Lsb0, prelude::BitArray};
    use decorum::{Real, N64};
    use num::FromPrimitive;

    pub fn optimal_specimen() -> Genome<10> {
        Genome::new(bitarr![u16, Lsb0; 1; 10])
    }

    pub fn encode_binary(pheonotype: f64) -> Genome<10> {
        let pheonotype = (pheonotype * 100.0) as u16;
        Genome::new(BitArray::new([pheonotype]))
    }

    pub fn decode_binary(genome: Genome<10>) -> N64 {
        let [pheonotype] = genome.into_inner();
        N64::from_u16(pheonotype).unwrap() / 100.0
    }

    pub fn fitness(pheonotype: N64) -> N64 {
        pheonotype.powi(2)
    }

    #[test]
    fn test_encode_binary() {
        use bitvec::{bitarr, prelude::Lsb0};
        let genome = encode_binary(0.0);
        assert_eq!(genome, Genome::new(bitarr![u16, Lsb0; 0; 10]));
        let genome = encode_binary(0.03);
        assert_eq!(
            genome,
            Genome::new(bitarr![u16, Lsb0; 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        );
        let genome = encode_binary(10.23);
        assert_eq!(
            genome,
            Genome::new(bitarr![u16, Lsb0; 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        );
    }

    #[test]
    fn test_decode_binary() {
        use bitvec::{bitarr, prelude::Lsb0};
        let genome = Genome::new(bitarr![u16, Lsb0; 0; 10]);
        assert_eq!(decode_binary(genome), 0.0);
        let genome = Genome::new(bitarr![u16, Lsb0; 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(decode_binary(genome), 0.03);
        let genome = Genome::new(bitarr![u16, Lsb0; 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(decode_binary(genome), 10.23);
    }
}

pub mod pow2 {
    use bitvec::{bitarr, order::Lsb0, prelude::BitArray};
    use decorum::N64;
    use num::FromPrimitive;

    use crate::Genome;

    pub fn optimal_specimen() -> Genome<10> {
        Genome::new(bitarr![u16, Lsb0; 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    }

    pub fn encode_binary(pheonotype: f64) -> Genome<10> {
        let pheonotype = ((pheonotype * 100.0) as i16 + 511) as u16;
        Genome::new(BitArray::new([pheonotype]))
    }

    pub fn decode_binary(genome: Genome<10>) -> N64 {
        let [pheonotype] = genome.into_inner();
        N64::from_u16(pheonotype).unwrap() / 100.0 - 511.0
    }

    pub fn fitness(pheonotype: N64) -> N64 {
        let _512 = N64::from(512.0);
        (_512 - pheonotype) * (_512 + pheonotype)
    }
}
