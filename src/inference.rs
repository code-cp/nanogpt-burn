use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};
use std::sync::Arc;

use crate::data::{TextGenerationBatcher, TextGenerationItem}; 
use crate::tokenizer::{SimpleTokenizer, Tokenizer};
use crate::training::ExperimentConfig;
use crate::model::TextGenerationModelConfig; 

pub fn infer<B: Backend> (
    device: B::Device, 
    artifact_dir: &str, 
    tokenizer: SimpleTokenizer,
) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file should present");

    let tokenizer = Arc::new(tokenizer); 

    // Initialize batcher for batching samples
    let batcher = Arc::new(TextGenerationBatcher::new(
        tokenizer.clone(),
        config.block_size,
    ));

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model weights");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = TextGenerationModelConfig::new(
        config.transformer,
        tokenizer.vocab_size(),
        config.block_size,
    )
    .init_with::<B>(record) // Initialize model with loaded weights
    .to_device(&device); // Move model to computation device\

    // Run inference on the given text samples
    println!("Running inference ...");
    let mut samples = TextGenerationItem::new("".to_string());
    let n_chars = 100; 
    for _ in 0..n_chars {
        let item = batcher.batch(samples.clone()); // Batch samples using the batcher
        let prediction = model.infer(item); // Get model predictions
    
        let logits = prediction.to_data(); // Convert prediction tensor to data
        let class_index = prediction.argmax(1).into_data().convert::<i32>().value[0];
    
        let new_char = tokenizer.untokenize(&vec![class_index]); 
        samples = TextGenerationItem::new(samples.text + new_char.as_str()); 

        // Print sample text, predicted logits and predicted class
        println!("{new_char}"); 
    }
}