use std::fs;

use burn::data::dataset::Dataset; 
use derive_new::new;

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String, 
}

pub struct TinyShakespeareDataset {
    dataset: String,
    block_size: usize,  
}

/// Implement dataset trait for custom dataset 
/// ref <https://docs.rs/burn-dataset/latest/burn_dataset/trait.Dataset.html>
impl Dataset<TextGenerationItem> for TinyShakespeareDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        if index >= self.len() {
            return None; 
        }

        let data = self.dataset
            .chars()
            .skip(index)
            .take(self.block_size)
            .collect::<String>(); 

        Some(TextGenerationItem::new(data)) 
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.block_size + 1 
    }
}

impl TinyShakespeareDataset {
    pub fn new_with_contents(contents: &str, split: &str, block_size: usize) -> Self {
        let (train, test) = contents.split_at(contents.len() * 9 / 10); 

        let dataset = match split {
            "train" => train, 
            "test" => test, 
            _ => panic!("{} is not train or test", split),
        }; 

        Self {
            dataset: String::from(dataset), 
            block_size, 
        }
    }
    
    pub fn new(data_file: &str, split: &str, batch_size: usize) -> Self {
        let contents = fs::read_to_string(data_file).unwrap(); 
        Self::new_with_contents(&contents, split, batch_size)
    } 

    pub fn train(data_file: &str, block_size: usize) -> Self {
        Self::new(data_file, "train", block_size) 
    }

    pub fn test(data_file: &str, block_size: usize) -> Self {
        Self::new(data_file, "test", block_size)
    }
}

#[cfg(test)]
mod test {
    use super::*; 
    use std::{fs, io::prelude::*, path::Path}; 

    #[test]
    pub fn check_length() {
        let dataset = TinyShakespeareDataset::new_with_contents(
            "0123456789", 
            "train", 
            4 
        ); 
        assert_eq!(5, dataset.len()); 
    }

    #[test]
    pub fn check_get() {
        let dataset = TinyShakespeareDataset::new_with_contents(
            "0123456789", 
            "train", 
            4 
        ); 
        assert_eq!(
            Some("01234".to_string()),
            dataset.get(0).map(|x| x.text)
        );
    }

    #[test]
    fn test_tiny_shakespeare_dataset() {
        let dataset_char = fs::read_to_string("./data/dataset.txt").expect("Should read dataset");
        let dataset = TinyShakespeareDataset {
            dataset: dataset_char,
            block_size: 8, 
        };

        println!("{:?}", dataset.get(0).unwrap()); 
        println!("{:?}", dataset.get(100_00000).unwrap()); 
    }
}