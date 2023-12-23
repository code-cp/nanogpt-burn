use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, ElementConversion, Int, Shape, Tensor},
};
use derive_new::new;
use std::sync::Arc;

use crate::tokenizer::Tokenizer; 
use crate::data::TextGenerationItem; 

#[derive(new)]
pub struct TextGenerationBatcher {
    tokenizer: Arc<dyn Tokenizer>, 
    block_size: usize, 
}

#[derive(Debug, Clone, new)]
pub struct TextGenerationBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>, 
}

#[derive(Debug, Clone, new)]
pub struct TrainingTextGenerationBatch<B: Backend> {
    /// Input tokens 
    pub tokens: Tensor<B, 2, Int>, 
    /// Expected outputs for the input tokens  
    pub targets: Tensor<B, 2, Int>, 
}

impl<B: Backend> Batcher<TextGenerationItem, TextGenerationBatch<B>> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> TextGenerationBatch<B> {
        let batch_size = items.len(); 
        let mut tokens = Tensor::zeros([batch_size, self.block_size+1], &B::Device::default()); 

        for i in 0..batch_size {
            let input = self.tokenizer.tokenize(&items[i].text); 
            // println!("TextGenerationBatcher input {:?}", input); 

            tokens = tokens.slice_assign(
                [i..i+1, 0..self.block_size+1], 
                Tensor::from_data(
                    Data::new(
                        input.into_iter().map(|e| (e as i64).elem()).collect(),
                        Shape::new([1, self.block_size+1]),
                    ),
                    &B::Device::default(),  
                ), 
            ); 
        }

        TextGenerationBatch {
            tokens,
        }
    }
}

impl<B: Backend> Batcher<TextGenerationItem, TrainingTextGenerationBatch<B>> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> TrainingTextGenerationBatch<B> {
        let item: TextGenerationBatch<B> = self.batch(items); 
        let [batch_size, block_size] = item.tokens.dims(); 
        // println!("batch size is {batch_size}"); 
        let block_size = block_size - 1; 

        let tokens = item 
            .tokens
            .clone()
            .slice([0..batch_size, 0..block_size]);
        // println!("tokens {:?}", tokens.dims()); 
        
        let targets = item
            .tokens
            .clone()
            .slice([0..batch_size, 1..block_size+1]);
        // println!("targets {:?}", targets.dims());

        TrainingTextGenerationBatch {
            tokens, 
            targets, 
        } 
    }
}