use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder, DefaultRecorder},
    tensor::backend::Backend,
    tensor::{activation, ElementConversion, Bool, Int, Tensor, Device},
};
use std::sync::Arc;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

use crate::data::{TextGenerationBatcher, TextGenerationItem}; 
use crate::tokenizer::{SimpleTokenizer, Tokenizer};
use crate::training::ExperimentConfig;
use crate::model::TextGenerationModelConfig; 

pub fn infer<B: Backend> (
    artifact_dir: &str, 
    tokenizer: SimpleTokenizer,
) -> String {
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
    let mut rng = rand::thread_rng();

    let initial_input = vec![" "; config.block_size+1].join(""); 
    let mut samples = TextGenerationItem::new(initial_input);
    // let n_chars = config.block_size;
    let n_chars = 1; 
    let mut output = String::new(); 
    for _ in 0..n_chars {
        // println!("samples {}", samples.text); 

        let item = batcher.batch(vec![samples.clone(); config.batch_size]); 
        let logits = model.infer(item); 
        // focus only on the last time step
        // shape is 1 x 1 x vocab size
        let logits = logits.slice([0..1, config.block_size-1..config.block_size, 0..tokenizer.vocab_size()]);
        let max_val = logits.clone().max().into_scalar().elem::<f32>(); 
        // println!("logits max val {:?}", max_val);
        let logits = logits.sub_scalar(max_val); 
        // println!("logits {:?}", logits.to_data());
        // let class_index = logits.argmax(2).into_data().convert::<i32>().value[0];

        let prob = activation::softmax(logits, 2);
        // println!("prob {:?}", prob.to_data());
        let mut probabilities: Vec<f32> = Vec::new(); 
        for val in prob.iter_dim(2) {
            probabilities.push(val.into_scalar().elem::<f32>()); 
        }
        // println!("probabilities {probabilities:?}"); 
        let weighted_index = WeightedIndex::new(&probabilities).unwrap();
        let class_index = weighted_index.sample(&mut rng);

        // println!("class_index {class_index}"); 
    
        let new_char = tokenizer.untokenize(&[class_index as usize]); 
        samples = TextGenerationItem::new(samples.text[1..].to_string() + new_char.as_str()); 

        // Print sample text, predicted logits and predicted class
        // avoid println which prints a new line 
        // print!("{new_char}"); 

        output += new_char.as_str(); 
    }

    output
}