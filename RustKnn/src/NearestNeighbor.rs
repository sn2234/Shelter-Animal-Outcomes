
use std::f64;

pub trait Knn {
    fn distance(&self, y : &Self) -> f64;
}

pub fn nearestNeighbor<T>(instances : Vec<T>, sample : T) -> T
    where T : Knn + Clone {
    //instances.iter().min_by_key(|x| sample.distance(x)).unwrap()
    let mut minInstanceIdx = 0;
    let mut minDistance = f64::INFINITY;
    
    for i in 0..instances.len() {
        let currentDistance = sample.distance(&instances[i]);
        if currentDistance < minDistance {
            minInstanceIdx = i;
            minDistance = currentDistance
        }
    }
    
    return instances[minInstanceIdx].clone()
}
