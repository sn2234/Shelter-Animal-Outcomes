
use csv;
use NearestNeighbor::Knn;
use std::f64;
use std::ops::Mul;

#[derive(RustcDecodable,Debug,Copy,Clone)]
pub struct Animal {
    pub AnimalID        : f64,
    pub Name            : f64,
    pub DateTime        : f64,
    pub OutcomeType     : usize,
    pub OutcomeSubtype  : usize,
    pub AnimalType      : f64,
    pub SexuponOutcome  : f64,
    pub AgeuponOutcome  : f64,
    pub Breed           : f64,
    pub Color           : f64
}

pub fn compareAnimals(x: &Animal, y: &Animal) -> bool {
    x.OutcomeType == y.OutcomeType
}

pub fn loadData(fileName : &str) -> csv::Result<Vec<Animal>> {
    csv::Reader::from_file(fileName)
                            .and_then(|x| Ok(x.has_headers(true)))
                            .and_then(|mut x| x.decode().collect::<csv::Result<Vec<Animal>>>())
}

fn sq<T>(x : T) -> T
where T: Mul<Output=T> + Copy {
    x*x
}

fn euclidianDistance(from: &Animal, to: &Animal) -> f64 {
        let sumSquares = sq(from.Name - to.Name) +
                         sq(from.DateTime - to.DateTime) +
                         sq(from.AnimalType - to.AnimalType) +
                         sq(from.SexuponOutcome - to.SexuponOutcome) +
                         sq(from.AgeuponOutcome - to.AgeuponOutcome) +
                         sq(from.Breed - to.Breed) +
                         sq(from.Color - to.Color);

        return (sumSquares as f64).sqrt();
}

fn manhattanDistance(from: &Animal, to: &Animal) -> f64 {
    let dist =  (from.Name - to.Name).abs() +
                (from.DateTime - to.DateTime).abs() +
                (from.AnimalType - to.AnimalType).abs() +
                (from.SexuponOutcome - to.SexuponOutcome).abs() +
                (from.AgeuponOutcome - to.AgeuponOutcome).abs() +
                (from.Breed - to.Breed).abs() +
                (from.Color - to.Color).abs();
    
    return dist;
}

fn minkowskyDistance(from: &Animal, to: &Animal) -> f64 {
    let q = 7.0;
    
    let distSum =   (from.Name - to.Name).abs().powf(q) +
                    (from.DateTime - to.DateTime).abs().powf(q) +
                    (from.AnimalType - to.AnimalType).abs().powf(q) +
                    (from.SexuponOutcome - to.SexuponOutcome).abs().powf(q) +
                    (from.AgeuponOutcome - to.AgeuponOutcome).abs().powf(q) +
                    (from.Breed - to.Breed).abs().powf(q) +
                    (from.Color - to.Color).abs().powf(q);
    
    return distSum.powf(1.0/q);
}

impl Knn for Animal {
    fn distance(&self, y : &Self) -> f64 {
        manhattanDistance(self, y)
    }
}
