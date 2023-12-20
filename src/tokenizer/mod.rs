mod simple; 
pub use simple::*; 

pub trait Tokenizer: Send + Sync {
    fn vocab_size(&self) -> usize; 
    fn tokenize(&self, string: &str) -> Vec<usize>;
    fn untokenize(&self, tokens: &[usize]) -> String; 
}
