use burn::{
    config::Config,
    data::{dataloader::{DataLoaderBuilder, DataLoader}, dataset::transform::SamplerDataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::Module,
    optim::{AdamConfig, Adam},
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CUDAMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use burn::data::dataset::Dataset; 
use std::sync::Arc; 

use crate::{gpt::TransformerDecoderConfig, tokenizer::{self, Tokenizer}, data::TextGenerationBatcher, model::TextGenerationModelConfig}; 
use crate::data::TextGenerationItem; 
use crate::tokenizer::SimpleTokenizer; 

#[derive(Config)]
pub struct ExperimentConfig {
    pub transformer: TransformerDecoderConfig, 
    pub optimizer: AdamConfig,
    pub batch_size: usize, 
    pub block_size: usize, 
    pub max_iters: usize,     
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device, 
    dataset_train: D, 
    dataset_test: D, 
    config: ExperimentConfig, 
    tokenizer: SimpleTokenizer,
    artifact_dir: &str, 
) {
    let tokenizer = Arc::new(tokenizer); 
    let batcher_train = TextGenerationBatcher::new(tokenizer.clone(), config.block_size); 
    let batcher_test = TextGenerationBatcher::new(tokenizer.clone(), config.block_size); 

    let model = TextGenerationModelConfig::new(
        config.transformer.clone(), 
        tokenizer.vocab_size(), 
        config.block_size, 
    ).init::<B>(&device); 

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 10_000)); 

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 1000)); 

    let accum = 6; // Effective batch size = 6 * 6 = 32.
    let optim = config.optimizer.init();
    let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
        .with_warmup_steps(6000)
        .with_model_size(config.transformer.n_embd)
        .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .grads_accumulation(accum)
        .num_epochs(config.max_iters)
        .build(model, optim, lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    DefaultRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}