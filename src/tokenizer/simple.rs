use super::Tokenizer; 
use std::collections::HashMap;

use std::{
    collections::HashSet,
    hash::{BuildHasher, Hasher},
};

/// There are only 65 chars, no need to use Google’s SwissTable design
#[derive(Debug, Clone, Default)]
struct CharHasher {
    c: char, 
}

impl Hasher for CharHasher {
    /// Returns the hash value for char 
    fn finish(&self) -> u64 {
        // hash function is just identity 
        self.c as _ 
    }

    /// Writes some data into this Hasher
    fn write(&mut self, bytes: &[u8]) {
        let (bytes, _) = bytes.split_at(std::mem::size_of::<char>());

        // Create a native endian integer value from its memory representation as a byte array in native endianness
        // char is 4 bytes in rust 
        let i = u32::from_ne_bytes(bytes.try_into().unwrap_or([0; 4])); 
        self.c = char::from_u32(i).unwrap_or('a')
    }
}

impl BuildHasher for CharHasher {
    type Hasher = Self;

    fn build_hasher(&self) -> Self::Hasher {
        Self::default()
    }
}

pub struct SimpleTokenizer {
    vocab_size: usize, 
    ch_to_int: HashMap<char, usize>,
    int_to_ch: HashMap<usize, char>, 
}

impl SimpleTokenizer {
    pub fn new(dataset: &str) -> Self {
        let mut chars = dataset
            .chars()
            .collect::<HashSet<char, CharHasher>>()
            .into_iter()
            .collect::<Vec<_>>(); 
        chars.sort();
        // println!("chars in tokenizer {chars:?}"); 
        
        let int_to_ch = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (i, *ch))
            .collect::<HashMap<usize, char>>(); 

        let ch_to_int = chars  
            .iter()
            .enumerate()
            .map(|(i, ch)| (*ch, i))
            .collect::<HashMap<char, usize>>(); 

        Self {
            vocab_size: chars.len(),
            int_to_ch, 
            ch_to_int, 
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size 
    }

    fn tokenize(&self, string: &str) -> Vec<usize> {
        string
            .chars()
            .map(|ch| self.ch_to_int.get(&ch).unwrap().clone())
            .collect() 
    }

    fn untokenize(&self, tokens: &[usize]) -> String {
        tokens 
            .iter()
            .map(|tkn| self.int_to_ch.get(tkn).unwrap().clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = SimpleTokenizer::new("hello world");
        let text = "hello"; 
        let tokens = tokenizer.tokenize(text);
        let decoded = tokenizer.untokenize(&tokens); 

        assert_eq!(decoded, text);  
    }
}
