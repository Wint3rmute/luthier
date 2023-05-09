use std::collections::HashMap;

use node_traits::{DspConnectible, DspNode};
use numpy::ndarray::{Array1, Dim};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

extern crate node_macro;
use node_macro::DspConnectibleDerive;

type NodeId = usize;
type InputId = usize;
type OutputId = usize;

#[derive(Debug)]
struct DspConnection {
    from_node: NodeId,
    from_output: OutputId,
    to_node: NodeId,
    to_input: InputId,
}

#[pyclass]
#[derive(Default, DspConnectibleDerive)]
struct Speaker {
    input_input: f64,
}

impl DspNode for Speaker {
    fn tick(&mut self) {}
}

#[pyclass]
#[derive(DspConnectibleDerive)]
struct SineOscillator {
    input_frequency: f64,
    input_modulation: f64,
    output_output: f64,

    state: usize,
}

// impl DspNode for Speaker {
//     fn num_inputs(&self) -> usize {
//         1
//     }

//     fn num_outputs(&self) -> usize {
//         0
//     }

//     fn input_names(&self) -> &[&str] {
//         &["input"][..]
//     }

//     fn output_names(&self) -> &[&str] {
//         &[][..]
//     }

//     fn set_input(&self, id: InputId, value: f64) {}

//     fn get_output(&self, id: OutputId) -> f64 {
//         0.0
//     }
// }

type Node = Box<dyn DspNode + Send>;

#[pyclass]
struct DspGraph {
    nodes: HashMap<NodeId, Box<dyn DspNode + Send>>,
    connections: Vec<DspConnection>,
    current_node_index: NodeId,
    speaker_node_id: NodeId,
}

impl Default for DspGraph {
    fn default() -> Self {
        let mut result = Self {
            nodes: HashMap::new(),
            connections: vec![],
            current_node_index: 0,
            speaker_node_id: 0,
        };

        result.speaker_node_id = result.add_node(Box::new(Speaker::default()));

        result
    }
}

impl DspGraph {
    fn get_next_node_index(&mut self) -> usize {
        self.current_node_index += 1;
        self.current_node_index
    }

    fn add_node(&mut self, node: Node) -> NodeId {
        let node_index = self.get_next_node_index();
        self.nodes.insert(node_index, node);
        node_index
    }

    fn get_node(&self, node_id: NodeId) -> &Node {
        self.nodes
            .get(&node_id)
            .unwrap_or_else(|| panic!("Node with id {} not found", node_id))
    }
}

#[pymethods]
impl DspGraph {
    #[new]
    fn new() -> Self {
        DspGraph::default()
    }

    fn nodes(&self) {
        println!("Co jest");
        println!("{:?}", self.connections);
    }

    fn tick(&mut self) -> f64 {
        for connection in self.connections.iter() {
            let output_node = &self.nodes[&connection.from_output];
            let value_on_output = output_node.get_output_by_index(connection.from_output);

            let input_node = self
                .nodes
                .get_mut(&connection.to_node)
                .unwrap_or_else(|| panic!("Node with id {} not found", connection.to_node));

            input_node.set_input_by_index(connection.to_input, value_on_output);
        }

        for (_node_id, node) in self.nodes.iter_mut() {
            node.tick();
        }

        self.get_node(self.speaker_node_id).get_input_by_index(0)
    }

    fn play<'py>(
        &mut self,
        num_samples: usize,
        py: Python<'py>,
    ) -> &'py PyArray<f64, Dim<[usize; 1]>> {
        let mut result = Array1::zeros(num_samples);

        for element in result.iter_mut() {
            *element = self.tick();
        }

        result.into_pyarray(py)
    }
}

#[pymodule]
fn luthier(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DspGraph>()?;
    m.add_class::<SineOscillator>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // struct SineOscillatorInputs {
    //     frequency: f64,
    //     modulation: f64,
    // }

    // struct SineOscillatorOutputs {
    //     output: f64,
    // }

    // #[derive(AnswerFn)]
    // struct SineOscillator {
    //     state: usize,

    //     input_frequency: f64,
    //     input_modulation: f64,
    //     output_output: f64,
    // }

    // #[test]
    // fn it_works() {
    //     let s = SineOscillator {
    //         state: 0,
    //         input_frequency: 0.0,
    //         input_modulation: 0.0,
    //         output_output: 0.0,
    //     };
    //     let names = SineOscillator::get_input_names();
    //     println!("{:?}", names);
    //     // assert_eq!(42, answer());
    //     //
    //     println!("{}", s.get_input_by_index(1));
    // }
}
