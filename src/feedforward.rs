use burn::nn::{Initializer, ReLU};
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

// use libm::sqrtf;

#[derive(Config)]
pub struct FeedForwardConfig {
    pub n_embd: usize, 
    /// The dropout rate. Default: 0.2
    #[config(default = 0.2)]
    pub dropout: f64, 
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/libm::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
} 

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            linear1: LinearConfig::new(
                self.n_embd, 
                4 * self.n_embd, 
            ).with_initializer(self.initializer.clone())
            .init(device), 
            linear2: LinearConfig::new(
                self.n_embd, 
                4 * self.n_embd, 
            ).with_initializer(self.initializer.clone())
            .init(device),
            dropout: DropoutConfig::new(self.dropout).init(), 
            relu: ReLU::new(), 
        }
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>, 
    linear2: Linear<B>, 
    dropout: Dropout, 
    relu: ReLU, 
}

impl<B: Backend> FeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear1.forward(input); 
        let x = self.relu.forward(x); 
        let x = self.dropout.forward(x); 
        self.linear2.forward(x)
    }
}
