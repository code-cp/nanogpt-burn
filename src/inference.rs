use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder, DefaultRecorder},
    tensor::backend::Backend,
};
use std::sync::Arc;

use crate::data::{TextGenerationBatcher, TextGenerationItem}; 
use crate::tokenizer::{SimpleTokenizer, Tokenizer};
use crate::training::ExperimentConfig;
use crate::model::TextGenerationModelConfig; 

pub fn infer<B: Backend> (
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
    let record = DefaultRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Should load trained model weights");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = TextGenerationModelConfig::new(
        config.transformer,
        tokenizer.vocab_size(),
        config.block_size,
    )
    .init_with::<B>(record); // Move model to computation device\

    // Run inference on the given text samples
    println!("Running inference ...");
    let initial_input = vec![" "; config.block_size+1].join(""); 
    let mut samples = TextGenerationItem::new(initial_input);
    let n_chars = config.block_size; 
    for _ in 0..n_chars {
        // println!("samples {}", samples.text); 

        let item = batcher.batch(vec![samples.clone(); config.batch_size]); 
        let logits = model.infer(item); 
        // focus only on the last time step
        // shape is 1 x 1 x vocab size
        let logits = logits.slice([0..1, config.block_size-1..config.block_size, 0..tokenizer.vocab_size()]);
        // println!("logits shape {:?}", logits.dims()); 
        let class_index = logits.argmax(2).into_data().convert::<i32>().value[0];
        // println!("class_index {class_index}"); 
    
        let new_char = tokenizer.untokenize(&[class_index as usize]); 
        samples = TextGenerationItem::new(samples.text[1..].to_string() + new_char.as_str()); 

        // Print sample text, predicted logits and predicted class
        // avoid println which prints a new line 
        print!("{new_char}"); 
    }
}