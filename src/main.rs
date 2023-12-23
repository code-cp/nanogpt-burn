use std::{fs, io::prelude::*, path::Path}; 
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use burn::optim::decay::WeightDecayConfig;

use nanogpt::training::{train, ExperimentConfig}; 
use nanogpt::gpt::TransformerDecoderConfig; 
use nanogpt::tokenizer::SimpleTokenizer; 
use nanogpt::data::TinyShakespeareDataset; 

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() {
    let config = ExperimentConfig::new(
        TransformerDecoderConfig::new(),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    let data_dir = "./data/dataset.txt"; 
    let dataset_char = fs::read_to_string(data_dir).expect("Should read dataset");
    let tokenizer = SimpleTokenizer::new(&dataset_char); 

    train::<Backend, TinyShakespeareDataset>(
        WgpuDevice::default(),
        TinyShakespeareDataset::train(data_dir, config.batch_size),
        TinyShakespeareDataset::test(data_dir, config.batch_size),
        config,
        tokenizer, 
        "/tmp",
    );
}
