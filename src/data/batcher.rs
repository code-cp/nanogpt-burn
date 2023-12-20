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
    context_size: usize, 
}

#[derive(Debug, Clone)]
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
        let mut tokens = Tensor::zeros([batch_size, self.context_size+1]); 

        for i in 0..batch_size {
            let input = self.tokenizer.tokenize(&items[i].text); 

            tokens = tokens.slice_assign(
                [i..i+1, 0..self.context_size+1], 
                Tensor::from_data(
                    Data::new(
                        input.into_iter().map(|e| (e as i64).elem()).collect(),
                        Shape::new([1, self.context_size+1]), 
                    )
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
        let [batch_size, context_size] = item.tokens.dims(); 
        let context_size = context_size - 1; 

        let tokens = item 
            .tokens
            .clone()
            .slice([0..batch_size, 0..context_size]);
        
        let targets = item
            .tokens
            .clone()
            .slice([0..batch_size, 1..context_size+1]);

        TrainingTextGenerationBatch {
            tokens, 
            targets, 
        } 
    }
}