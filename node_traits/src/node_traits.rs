pub trait DspConnectible {
    fn get_input_names(&self) -> Vec<String>;
    fn get_output_names(&self) -> Vec<String>;
    fn get_input_by_index(&self, index: usize) -> f64;
    fn get_output_by_index(&self, index: usize) -> f64;
    fn set_input_by_index(&mut self, index: usize, value: f64);

    fn get_index_of_input(&self, input_name: &str) -> Option<usize> {
        for (index, name) in self.get_input_names().iter().enumerate() {
            if &input_name == name {
                return Some(index);
            }
        }
        None
    }

    fn get_index_of_output(&self, output_name: &str) -> Option<usize> {
        for (index, name) in self.get_output_names().iter().enumerate() {
            if &output_name == name {
                return Some(index);
            }
        }
        None
    }
}

pub trait DspNode: DspConnectible {
    fn tick(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // let result = add(2, 2);
        // assert_eq!(result, 4);
    }
}
