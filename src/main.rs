use std::{fs, io::prelude::*, path::Path}; 

use nanogpt::tokenizer::*; 
use nanogpt::data::*; 

fn main() {
    let dataset_char = fs::read_to_string("./data/dataset.txt").expect("Should read dataset");
    let tokenizer = SimpleTokenizer::new(&dataset_char); 
    let dataset = tokenizer.tokenize(&dataset_char);  
}
