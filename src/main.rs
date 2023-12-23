use std::{fs, io::prelude::*, path::Path}; 
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use burn::optim::decay::WeightDecayConfig;
// use burn::backend::{libtorch::LibTorchDevice, LibTorch};

use nanogpt::training::{train, ExperimentConfig}; 
use nanogpt::gpt::TransformerDecoderConfig; 
use nanogpt::tokenizer::SimpleTokenizer; 
use nanogpt::data::TinyShakespeareDataset; 
use nanogpt::inference::infer; 

// #[cfg(feature = "f16")]
// type Elem = burn::tensor::f16;
// #[cfg(not(feature = "f16"))]
// type Elem = f32;

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;
// type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

fn main() {
    let batch_size = 16; 
    let block_size = 100; 
    let max_iters = 10; 

    let config = ExperimentConfig::new(
        TransformerDecoderConfig::new(
           batch_size, 
           block_size,  
        ),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
        batch_size, 
        block_size, 
        max_iters, 
    );

    let data_dir = "./data/dataset.txt"; 
    let dataset_char = fs::read_to_string(data_dir).expect("Should read dataset");
    let tokenizer = SimpleTokenizer::new(&dataset_char); 
    let artifact_dir = "./tmp"; 

    // train::<Backend, TinyShakespeareDataset>(
    //     WgpuDevice::default(),
    //     // LibTorchDevice::Cuda(0),
    //     // if cfg!(target_os = "macos") {
    //     //     burn::tensor::Device::<Backend>::Mps
    //     // } else {
    //     //     burn::tensor::Device::<Backend>::Cuda(0)
    //     // },
    //     TinyShakespeareDataset::train(data_dir, config.batch_size),
    //     TinyShakespeareDataset::test(data_dir, config.batch_size),
    //     config,
    //     tokenizer, 
    //     artifact_dir,
    // );

    let tokenizer = SimpleTokenizer::new(&dataset_char); 
    infer::<Backend>(
        artifact_dir,
        tokenizer, 
    );  
}
