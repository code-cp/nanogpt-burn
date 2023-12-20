use burn::data::dataset::Dataset; 
use derive_new::new;

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String, 
}

pub struct TinyShakespeareDataset {
    dataset: String,
    context_size: usize,  
}

/// Implement dataset trait for custom dataset 
/// ref <https://docs.rs/burn-dataset/latest/burn_dataset/trait.Dataset.html>
impl Dataset<TextGenerationItem> for TinyShakespeareDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        let start_index = index; 
        let end_index = (start_index + self.context_size + 1) % self.len(); 

        let data = if start_index < end_index {
            self.dataset[start_index..end_index].to_string()
        } else {
            let mut wrapped_chars = String::new(); 
            wrapped_chars.push_str(&self.dataset[start_index..]); 
            wrapped_chars.push_str(&self.dataset[..end_index]); 
            wrapped_chars
        };

        Some(TextGenerationItem::new(data)) 
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}