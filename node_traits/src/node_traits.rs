pub trait DspConnectible {
    fn get_input_names(&self) -> Vec<String>;
    fn get_output_names(&self) -> Vec<String>;
    fn get_input_by_index(&self, index: usize) -> f64;
    fn get_output_by_index(&self, index: usize) -> f64;
    fn set_input_by_index(&mut self, index: usize, value: f64);
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
