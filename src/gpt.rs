use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Initializer, 
    },
    tensor::{backend::Backend, Tensor},
};

use crate::feedforward::{FeedForwardConfig, FeedForward}; 
use crate::attention::{MultiHeadAttention, MultiHeadAttentionConfig, HeadConfig, Head}; 

#[derive(Config)]
pub struct TransformerDecoderConfig {
    #[config(default=64)]
    pub batch_size: usize,
    /// Size of context length
    #[config(default=256)]  
    pub block_size: usize,  
    /// Size of the model 
    #[config(default=384)] 
    pub n_embd: usize, 
    /// Number of heads 
    #[config(default=6)] 
    pub n_head: usize, 
    /// Number of block 
    #[config(default=6)] 
    pub n_layer: usize, 
    /// The dropout rate. Default: 0.2
    #[config(default = 0.2)]
    dropout: f64, 
}

impl TransformerDecoderConfig {
    pub fn init<B: Backend> (&self, device: &B::Device) -> TransformerDecoder<B> {
        let blocks = (0..self.n_layer)
            .map(|_| TransformerDecoderBlock::new(self, device))
            .collect::<Vec<_>>(); 

        TransformerDecoder { blocks }
    }
}

#[derive(Module, Debug)]
pub struct TransformerDecoderBlock<B: Backend> {
    multi_head_attn: MultiHeadAttention<B>, 
    feed_forward: FeedForward<B>, 
    ln1: LayerNorm<B>, 
    ln2: LayerNorm<B>, 
    dropout: Dropout, 
}

impl<B: Backend> TransformerDecoderBlock<B> {
    fn new(config: &TransformerDecoderConfig, device: &B::Device) -> Self {
        let ln1 = LayerNormConfig::new(config.n_embd).init(device); 
        let ln2 = LayerNormConfig::new(config.n_embd).init(device); 
        let dropout = DropoutConfig::new(config.dropout).init(); 
        
        let feedforward_config = FeedForwardConfig::new(
            config.n_embd, 
            config.dropout,     
        );
        let feed_forward = feedforward_config.init(device);  

        let head_size = config.n_embd / config.n_head;
        // println!("head size {:?}", head_size); 

        let mha_config = MultiHeadAttentionConfig::new(
            config.n_layer,
            config.n_head, 
            head_size, 
            config.n_embd, 
            config.dropout, 
        );
        let head_config = HeadConfig::new(
            config.batch_size, 
            config.block_size, 
            config.n_embd, 
            head_size,
            config.dropout,      
        );
        let multi_head_attn = mha_config.init(device, &head_config); 

        Self {
            ln1,
            ln2,  
            dropout, 
            feed_forward,
            multi_head_attn, 
        }
    }
}

impl<B: Backend> TransformerDecoderBlock<B> {
    pub fn forward(&self, input: Tensor::<B, 3>) -> Tensor::<B, 3> {
        let x1 = self.ln1.forward(input); 
        let x2 = self.multi_head_attn.forward(x1.clone()); 
        let x3 = x1 + x2; 
        let x4 = self.ln2.forward(x3); 
        let x5 = self.feed_forward.forward(x4.clone());
        let x6 = x4 + x5;  
        x6
    }
}

#[derive(Module, Debug)]
pub struct TransformerDecoder<B: Backend> {
    blocks: Vec<TransformerDecoderBlock<B>>, 
}

impl<B: Backend> TransformerDecoder<B> {
    pub fn forward(&self, x: Tensor::<B, 3>) -> Tensor::<B, 3> {
        let mut x = x; 
        for layer in self.blocks.iter() {
            x = layer.forward(x); 
        }
        x 
    }
}


