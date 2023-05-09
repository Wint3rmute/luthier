pub type NodeId = usize;
pub type InputId = usize;
pub type OutputId = usize;

pub type Node = Box<dyn DspNode + Send>;

pub trait DspConnectible {
    fn node_name(&self) -> &str;
    fn get_input_names(&self) -> Vec<String>;
    fn get_output_names(&self) -> Vec<String>;
    fn get_input_by_index(&self, index: InputId) -> f64;
    fn get_output_by_index(&self, index: OutputId) -> f64;
    fn set_input_by_index(&mut self, index: InputId, value: f64);

    fn get_index_of_input(&self, input_name: &str) -> Option<InputId> {
        for (index, name) in self.get_input_names().iter().enumerate() {
            if &input_name == name {
                return Some(index);
            }
        }
        None
    }

    fn get_index_of_output(&self, output_name: &str) -> Option<OutputId> {
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
