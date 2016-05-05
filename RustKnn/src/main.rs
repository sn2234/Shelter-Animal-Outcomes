#![allow(non_snake_case,dead_code,non_camel_case_types,non_upper_case_globals)]
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate csv;
extern crate rustc_serialize;
extern crate simple_parallel;

mod DataModel;
mod NearestNeighbor;

use std::collections::BTreeMap;

//set RUST_LOG=RustKnn=debug

fn populateFromVote(instance: &DataModel::Animal, neighbors: &[DataModel::Animal]) -> DataModel::Animal {
    
    let mut outcomeMap = BTreeMap::new();
    
    for x in neighbors {
        let outcome = x.OutcomeType;
        
        if outcomeMap.contains_key(&outcome) {
            if let Some(curr) = outcomeMap.get_mut(&outcome) {
                *curr = *curr + 1;
            }
        } else {
            outcomeMap.insert(outcome, 1);
        }
    }
    
    let newOutcome = *(outcomeMap.iter().max_by_key(|x| x.1).unwrap().0);
    
    return DataModel::Animal { OutcomeType: newOutcome, .. *instance};
}

fn testKNearestNeighbors() {
    
    let k = 500;
    println!("Testing k-neighbors prediction with k: {}", k);
    
    let trainData = DataModel::loadData("..\\processed_train.csv").unwrap();
    println!("Loaded training data: {} elements", trainData.len());
    
    let cvData = DataModel::loadData("..\\processed_cv.csv").unwrap();
    println!("Loaded CV data: {} elements", cvData.len());

    let mut correctCount = 0;
    
    for testVal in &cvData {
        let neighbors = NearestNeighbor::kNearestNeighbors(k, &trainData, testVal);
        let updated = populateFromVote(&testVal, &neighbors);
        if updated.OutcomeType == testVal.OutcomeType {
            correctCount += 1;
        }
    }
    
    println!("Accuracy: {}", (correctCount as f64)/(cvData.len() as f64));
}

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
    testKNearestNeighbors();
/*
    let a = DataModel::loadData("..\\processed_train.csv").unwrap();
    
    println!("Loaded data: {} elements, first row: {:?}", a.len(), a[0]);
    
    let nn = NearestNeighbor::nearestNeighbor(&a[1..], &a[0]);
    
    println!("It's nearest neighbor is: {:?}", nn);
*/
}
