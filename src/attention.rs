use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::{
    config::Config,
    module::Module,
    tensor::{activation, backend::Backend, Bool, Int, Tensor},
};
// use libm::sqrtf;

#[derive(Config)]
pub struct HeadConfig {
    batch_size: usize,
    /// Context size
    block_size: usize,
    n_embd: usize,
    head_size: usize,
    dropout: f64,
    /// The type of function used to initialize neural network parameters
    /// NOTE, the format below cannot be changed, otherwise Config derive will have error 
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/libm::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl HeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Head<B> {
        // compute the weight matrix
        let tril: Tensor<B, 3, Int> =
            Tensor::ones([self.batch_size, self.block_size, self.block_size], device).tril(-1);
        let tril = tril.equal_elem(0);

        Head {
            key: LinearConfig::new(self.n_embd, self.head_size)
                .with_initializer(self.initializer.clone())
                .init(device),
            query: LinearConfig::new(self.n_embd, self.head_size)
                .with_initializer(self.initializer.clone())
                .init(device),
            value: LinearConfig::new(self.n_embd, self.head_size)
                .with_initializer(self.initializer.clone())
                .init(device),
            tril,
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: HeadRecord<B>) -> Head<B> {
        // compute the weight matrix
        let tril: Tensor<B, 3, Int> = Tensor::ones(
            [self.batch_size, self.block_size, self.block_size],
            &B::Device::default(),
        )
        .tril(-1);
        let tril = tril.equal_elem(0);

        Head {
            key: LinearConfig::new(self.n_embd, self.head_size).init_with(record.key),
            query: LinearConfig::new(self.n_embd, self.head_size).init_with(record.query),
            value: LinearConfig::new(self.n_embd, self.head_size).init_with(record.value),
            tril,
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Head<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    tril: Tensor<B, 3, Bool>,
    dropout: Dropout,
}

impl<B: Backend> Head<B> {
    /// Single head attention
    /// input of size (batch, time-step, channels)
    /// output of size (batch, time-step, head size)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // (B,T,hs)
        let k = self.key.forward(x.clone());
        // (B,T,hs)
        let q = self.query.forward(x.clone());
        // println!("q size {:?}", q.dims());
        // (B, T, hs) @ (B, hs, T) -> (B, T, T)
        let d = (k.dims()[2] as f32).sqrt();
        let kt = k.transpose();
        // println!("kt size {:?}", kt.dims());
        // NOTE, do NOT use *, which is elementwise multiplication
        let wei = q.matmul(kt) / d;
        // println!("wei size {:?}", wei.dims());
        // (B, T, T)
        // ref https://docs.rs/burn/0.9.0/burn/tensor/struct.Tensor.html#method.mask_fill
        // A value too low might result in NaN
        let wei = wei.mask_fill(self.tril.clone(), -1.0e4);
        // (B, T, T)
        // ref https://docs.rs/burn/0.9.0/burn/tensor/activation/fn.softmax.html
        let wei = activation::softmax(wei, 2);
        let wei = self.dropout.forward(wei);
        // (B,T,hs)
        let v = self.value.forward(x);
        // (B, T, T) @ (B, T, hs) -> (B, T, hs)
        let out = wei.matmul(v);
        out
    }
}

#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    pub n_layer: usize,
    pub n_head: usize,
    pub head_size: usize,
    pub n_embd: usize,
    pub dropout: f64,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        head_config: &HeadConfig,
    ) -> MultiHeadAttention<B> {
        let layers = (0..self.n_layer)
            .map(|_| head_config.init(device))
            .collect::<Vec<_>>();

        MultiHeadAttention {
            proj: LinearConfig::new(self.head_size * self.n_head, self.n_embd).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            heads: layers,
        }
    }

    pub fn init_with<B: Backend>(
        &self,
        head_config: &HeadConfig,
        record: MultiHeadAttentionRecord<B>,
    ) -> MultiHeadAttention<B> {
        let heads = record
            .heads
            .into_iter()
            .map(|record| head_config.init_with(record))
            .collect();

        MultiHeadAttention {
            proj: LinearConfig::new(self.head_size * self.n_head, self.n_embd)
                .init_with(record.proj),
            dropout: DropoutConfig::new(self.dropout).init(),
            heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    proj: Linear<B>,
    dropout: Dropout,
    heads: Vec<Head<B>>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut inputs = Vec::new();
        for head in self.heads.iter() {
            inputs.push(head.forward(x.clone()));
        }
        let x = Tensor::cat(inputs, 2);
        let x = self.proj.forward(x);
        let x = self.dropout.forward(x);
        x
    }
}
