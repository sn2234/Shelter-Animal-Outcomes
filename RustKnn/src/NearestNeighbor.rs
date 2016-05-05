
use std::collections::BinaryHeap;
use std::f64;
use std::cmp::Ord;
use std::cmp::Ordering;
use std::cmp::Eq;
use std::cmp::PartialOrd;
use std::cmp::PartialEq;

pub trait Knn {
    fn distance(&self, y : &Self) -> f64;
}

pub fn nearestNeighbor<T>(instances : &[T], sample : &T) -> T
    where T : Knn + Clone
{
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

struct Neighbor<'a, T:'a> {
    inst : &'a T,
    dist : f64
}

impl<'a, T> Ord for Neighbor<'a, T> {
    fn cmp(& self, other: & Self) -> Ordering {
        if self.dist > other.dist {
            Ordering::Greater
        } else if self.dist < other.dist {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }
}

impl<'a, T> PartialOrd for Neighbor<'a, T> {
    fn partial_cmp(& self, other: & Self) -> Option<Ordering> {
        return self.dist.partial_cmp(&other.dist);
    }
}

impl<'a, T> Eq for Neighbor<'a, T> {
    
}

impl<'a, T> PartialEq for Neighbor<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        return self.dist.eq(&other.dist);
    }
}


pub fn kNearestNeighbors<T>(k : usize, instances : &[T], sample : &T) -> Vec<T>
    where T : Knn + Clone
{
    let mut nHeap = BinaryHeap::with_capacity(k);
    
    for i in instances {
        nHeap.push(Neighbor { inst : i, dist : sample.distance(i) });
        
        if nHeap.len() > k {
            nHeap.pop();
        }
    }
    
    return nHeap.iter().map(|x| x.inst.clone()).collect::<Vec<_>>();
}
