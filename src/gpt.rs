use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, Tensor},
};

use crate::feedforward::{FeedForwardConfig, FeedForward}; 

#[derive(Config)]
pub struct TransformerDecoderConfig {
    /// Size of the model 
    pub n_embd: usize, 
    /// Number of heads 
    pub n_head: usize, 
    /// Number of block 
    pub n_layers: usize, 
}

impl TransformerDecoderConfig {
    pub fn init<B: Backend> (&self, device: &B::Device) -> TransformerDecoder<B> {
        let blocks = (0..self.n_layers)
            .map(|_| TransformerDecoderBlock::new(self, device))
            .collect::<Vec<_>>(); 

        TransformerDecoder { blocks }
    }
}

#[derive(Module, Debug)]
pub struct TransformerDecoderBlock<B: Backend> {
    ln1: LayerNorm<B>, 
    ln2: LayerNorm<B>, 
}

impl<B: Backend> TransformerDecoderBlock<B> {
    fn new(config: &TransformerDecoderConfig, device: &B::Device) -> Self {
        let ln1 = LayerNormConfig::new(config.n_embd).init(device); 
        let ln2 = LayerNormConfig::new(config.n_embd).init(device); 
        
        Self {
            ln1,
            ln2,  
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerDecoder<B: Backend> {
    blocks: Vec<TransformerDecoderBlock<B>>, 
}


