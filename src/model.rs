use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig,
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::backend::{AutodiffBackend, Backend},
    tensor::Tensor,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::TrainingTextGenerationBatch;
use crate::gpt::{TransformerDecoderConfig, TransformerDecoder}; 

#[derive(Config)]
pub struct TextGenerationModelConfig {
    transformer: TransformerDecoderConfig, 
    vocab_size: usize, 
    block_size: usize, 
}

impl TextGenerationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextGenerationModel<B> {
        let lm_head = LinearConfig::new(self.transformer.n_embd, self.vocab_size).init(device); 
        let transformer = self.transformer.init(device); 
        let embedding_token = 
            EmbeddingConfig::new(self.vocab_size, self.transformer.n_embd).init(device); 
        let embedding_pos = 
            EmbeddingConfig::new(self.block_size, self.transformer.n_embd).init(device); 

        TextGenerationModel {
            transformer, 
            embedding_token, 
            embedding_pos, 
            lm_head, 
            vocab_size: self.vocab_size, 
            block_size: self.block_size, 
        } 
    }

    /// Initializes a model with provided weights
    pub fn init_with<B: Backend>(
        &self,
        record: TextGenerationModelRecord<B>,
    ) -> TextGenerationModel<B> {
        let lm_head = LinearConfig::new(self.transformer.n_embd, self.vocab_size).init_with(record.lm_head); 
        let transformer = self.transformer.init_with(record.transformer); 
        let embedding_token = 
            EmbeddingConfig::new(self.vocab_size, self.transformer.n_embd).init_with(record.embedding_token); 
        let embedding_pos = 
            EmbeddingConfig::new(self.block_size, self.transformer.n_embd).init_with(record.embedding_pos); 

        TextGenerationModel {
            transformer, 
            embedding_token, 
            embedding_pos, 
            lm_head, 
            vocab_size: self.vocab_size, 
            block_size: self.block_size, 
        } 
    }
}

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    transformer: TransformerDecoder<B>, 
    embedding_token: Embedding<B>, 
    embedding_pos: Embedding<B>, 
    lm_head: Linear<B>, 
    vocab_size: usize, 
    block_size: usize, 
} 

impl<B: Backend> TextGenerationModel<B> {
    pub fn forward_training(
        &self, 
        item: TrainingTextGenerationBatch<B>, 
    ) -> ClassificationOutput<B> {
        let [batch_size, block_size] = item.tokens.dims();
        let device = &self.devices()[0];   

        let inputs = item.tokens.to_device(device); 
        let targets = item.targets.to_device(device); 

        // batch size x context size 
        let index_positions = Tensor::arange(0..block_size, device)
        .reshape([1, block_size])
        .repeat(0, batch_size); 

        let embedding_positions = self.embedding_pos.forward(index_positions); 
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = embedding_positions + embedding_tokens; 

        let encoded = self.transformer.forward(embedding);
        let output = self.lm_head.forward(encoded);
        let output_flatten = output.reshape([batch_size * block_size, self.vocab_size]); 
        let targets_flatten = targets.reshape([batch_size * block_size]); 

        let loss = CrossEntropyLossConfig::new().init(); 
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone()); 

        ClassificationOutput {
            loss, 
            output: output_flatten, 
            targets: targets_flatten, 
        }
    }

    pub fn infer(
        &self, 
        item: TrainingTextGenerationBatch<B>, 
    ) -> Tensor<B, 3> {
        let [batch_size, block_size] = item.tokens.dims();
        let device = &self.devices()[0];   

        let inputs = item.tokens.to_device(device); 

        // batch size x context size 
        let index_positions = Tensor::arange(0..block_size, device)
        .reshape([1, block_size])
        .repeat(0, batch_size); 

        let embedding_positions = self.embedding_pos.forward(index_positions); 
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = embedding_positions + embedding_tokens; 

        let encoded = self.transformer.forward(embedding);
        let output = self.lm_head.forward(encoded);
        output 
    }
}

impl<B: AutodiffBackend> TrainStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>> for TextGenerationModel<B> {
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item); 
        let grads = item.loss.backward(); 

        TrainOutput::new(self, grads, item)
    }
} 

impl<B: Backend> ValidStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>> for TextGenerationModel<B> {
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}