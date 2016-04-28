#![allow(non_snake_case,dead_code,non_camel_case_types,non_upper_case_globals)]
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate csv;
extern crate rustc_serialize;

mod DataModel;
//set RUST_LOG=RustKnn=debug

fn main() {
    env_logger::init().unwrap();
    info!("Test log");

    let a = DataModel::loadData("..\\processed_train.csv");
    
    println!("Loaded data {:?}", a[0]);
}
