use core::fmt;

use decorum::N64;

use crate::{EvaluatedGenome, Genome, RunState};

pub fn evaluate(state: &mut RunState, population: Vec<Genome>) -> Vec<EvaluatedGenome> {
    fn res(
        population: Vec<Genome>,
        fitness_fn: impl Fn(Genome) -> N64,
    ) -> Vec<EvaluatedGenome> {
        population
            .into_iter()
            .map(|genome| EvaluatedGenome {
                fitness: fitness_fn(genome),
                genome,
            })
            .collect()
    }

    match state.config.ty {
        AlgoType::HasPhoenotype(algo, encoding) => {
            let (decode_genome, decode_phenotype, fitness) = phenotype_fns(algo, encoding);
            res(population, |genome| {
                fitness(decode_phenotype(decode_genome(genome)))
            })
        }
        AlgoType::BinaryOnly(BinaryAlgo::FConst) => res(population, |_| 100.0.into()),
        AlgoType::BinaryOnly(BinaryAlgo::FHD { sigma }) => {
            res(population, |genome| fhd::fitness(sigma, genome))
        }
    }
}

fn phenotype_fns(
    algo: PheonotypeAlgo,
    encoding: GenomeEncoding,
) -> (fn(Genome) -> Genome, fn(Genome) -> N64, fn(N64) -> N64) {
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

pub fn gray_to_binary(mut genome: Genome) -> Genome {
    for i in 1..genome.len() {
        let value = genome[i - 1] ^ genome[i];
        genome.set(i, value);
    }
    genome
}

pub fn binary_to_gray(mut genome: Genome) -> Genome {
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
    genome: Genome,
) -> PheonotypeEvaluation {
    let (decode_genome, decode_phenotype, fitness) = phenotype_fns(algo, encoding);
    let pheonotype = decode_phenotype(decode_genome(genome));
    PheonotypeEvaluation {
        fitness: fitness(pheonotype),
        pheonotype,
    }
}

pub fn optimal_binary_specimen(algo: BinaryAlgo) -> Option<Genome> {
    match algo {
        BinaryAlgo::FConst => None,
        BinaryAlgo::FHD { .. } => Some(fhd::optimal_specimen()),
    }
}

pub fn optimal_pheonotype_specimen(algo: PheonotypeAlgo) -> Genome {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgoType {
    BinaryOnly(BinaryAlgo),
    HasPhoenotype(PheonotypeAlgo, GenomeEncoding),
}

impl fmt::Display for AlgoType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlgoType::BinaryOnly(algo) => algo.fmt(f),
            AlgoType::HasPhoenotype(algo, encoding) => write!(f, "{algo}-{encoding}"),
        }
    }
}

pub mod fhd {
    use super::Genome;
    use bitvec::{order::Lsb0, bitarr};
    use decorum::N64;
    use num::FromPrimitive;

    pub const GENE_LENGTH: usize = 100;

    pub fn optimal_specimen() -> Genome {
        Genome::Hundo(bitarr![u16, Lsb0; 0; 100])
    }

    pub fn fitness(sigma: N64, genome: Genome) -> N64 {
        let zeros = N64::from_usize(genome.count_zeros()).unwrap();
        let len = N64::from_usize(genome.len()).unwrap();
        len - zeros + zeros * sigma
    }
}

pub mod pow1 {
    use crate::GenomeLength;

    use super::Genome;
    use bitvec::{order::Lsb0, bitarr, prelude::BitArray};
    use decorum::{Real, N64};
    use num::FromPrimitive;

    pub const GENE_LENGTH: GenomeLength = GenomeLength::Ten;

    pub fn optimal_specimen() -> Genome {
        Genome::Ten(bitarr![u16, Lsb0; 1; 10])
    }

    pub fn encode_binary(pheonotype: f64) -> Genome {
        let pheonotype = (pheonotype * 100.0) as u16;
        Genome::Ten(BitArray::new([pheonotype]))
    }

    pub fn decode_binary(genome: Genome) -> N64 {
        let Genome::Ten(genome) = genome else { panic!("wrong genome length") };
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
        assert_eq!(genome, Genome::Ten(bitarr![u16, Lsb0; 0; 10]));
        let genome = encode_binary(0.03);
        assert_eq!(genome, Genome::Ten(bitarr![u16, Lsb0; 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]));
        let genome = encode_binary(10.23);
        assert_eq!(genome, Genome::Ten(bitarr![u16, Lsb0; 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    }

    #[test]
    fn test_decode_binary() {
        use bitvec::{bitarr, prelude::Lsb0};
        let genome = Genome::Ten(bitarr![u16, Lsb0; 0; 10]);
        assert_eq!(decode_binary(genome), 0.0);
        let genome = Genome::Ten(bitarr![u16, Lsb0; 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(decode_binary(genome), 0.03);
        let genome = Genome::Ten(bitarr![u16, Lsb0; 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(decode_binary(genome), 10.23);
    }
}

pub mod pow2 {
    use bitvec::{order::Lsb0, bitarr, prelude::BitArray};
    use decorum::N64;
    use num::FromPrimitive;

    use crate::Genome;

    pub const GENE_LENGTH: usize = 10;

    pub fn optimal_specimen() -> Genome {
        Genome::Ten(bitarr![u16, Lsb0; 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    }

    pub fn encode_binary(pheonotype: f64) -> Genome {
        let pheonotype = ((pheonotype * 100.0) as i16 + 511) as u16;
        Genome::Ten(BitArray::new([pheonotype]))
    }

    pub fn decode_binary(genome: Genome) -> N64 {
        let Genome::Ten(genome) = genome else { panic!("wrong genome length") };
        let [pheonotype] = genome.into_inner();
        N64::from_u16(pheonotype).unwrap() / 100.0 - 511.0
    }

    pub fn fitness(pheonotype: N64) -> N64 {
        let _512 = N64::from(512.0);
        (_512 - pheonotype) * (_512 + pheonotype)
    }
}
