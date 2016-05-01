
use csv;
use NearestNeighbor::Knn;
use std::f64;
use std::ops::Mul;

#[derive(RustcDecodable,Debug,Copy,Clone)]
pub struct Animal {
    AnimalID        : f64,
    Name            : f64,
    DateTime        : f64,
    OutcomeType     : f64,
    OutcomeSubtype  : f64,
    AnimalType      : f64,
    SexuponOutcome  : f64,
    AgeuponOutcome  : f64,
    Breed           : f64,
    Color           : f64
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

impl Knn for Animal {
    fn distance(&self, y : &Self) -> f64 {
        let sumSquares = sq(self.Name - y.Name) +
                         sq(self.DateTime - y.DateTime) +
                         sq(self.AnimalType - y.AnimalType) +
                         sq(self.SexuponOutcome - y.SexuponOutcome) +
                         sq(self.AgeuponOutcome - y.AgeuponOutcome) +
                         sq(self.Breed - y.Breed) +
                         sq(self.Color - y.Color);

        return (sumSquares as f64).sqrt();
    }
}
