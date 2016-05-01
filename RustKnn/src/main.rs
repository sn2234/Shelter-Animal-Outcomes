#![allow(non_snake_case,dead_code,non_camel_case_types,non_upper_case_globals)]
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate csv;
extern crate rustc_serialize;

mod DataModel;
mod NearestNeighbor;
//set RUST_LOG=RustKnn=debug

fn testSingleNeighbor() {
    println!("Testing single-neighbor prediction");
    
    let trainData = DataModel::loadData("..\\processed_train.csv").unwrap();
    println!("Loaded training data: {} elements", trainData.len());
    
    let cvData = DataModel::loadData("..\\processed_cv.csv").unwrap();
    println!("Loaded CV data: {} elements", cvData.len());
    
    let numberOfHits = cvData.iter().map(|x| {
        let nn = NearestNeighbor::nearestNeighbor(&trainData, &x);
        if DataModel::compareAnimals(&nn, &x) {
            1
        } else {
            0
        }
    }).fold(0, |acc, x| acc + x);
    
    println!("Accuracy: {}", (numberOfHits as f64)/(cvData.len() as f64)); 
}

fn main() {
    env_logger::init().unwrap();
    info!("Test log");

    testSingleNeighbor();
/*
    let a = DataModel::loadData("..\\processed_train.csv").unwrap();
    
    println!("Loaded data: {} elements, first row: {:?}", a.len(), a[0]);
    
    let nn = NearestNeighbor::nearestNeighbor(&a[1..], &a[0]);
    
    println!("It's nearest neighbor is: {:?}", nn);
*/
}
