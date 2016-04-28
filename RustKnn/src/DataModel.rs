
use csv;
//use rustc_serialize;

#[derive(RustcDecodable,Debug)]
pub struct Animal {
    AnimalID        : u32,
    Name            : u32,
    DateTime        : u32,
    OutcomeType     : u32,
    OutcomeSubtype  : u32,
    AnimalType      : u32,
    SexuponOutcome  : u32,
    AgeuponOutcome  : u32,
    Breed           : u32,
    Color           : u32
}

pub fn loadData(fileName : &str) -> Vec<Animal> {
    let mut rdr = csv::Reader::from_file(fileName)
                            .unwrap()
                            .has_headers(true);
    
    rdr.decode().collect::<csv::Result<Vec<Animal>>>().unwrap()
}
