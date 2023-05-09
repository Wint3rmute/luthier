use std::collections::HashMap;

use node_traits::{DspConnectible, DspNode, InputId, Node, NodeId, OutputId};
use numpy::ndarray::{Array1, Dim};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

extern crate node_macro;
use node_macro::DspConnectibleDerive;

const SAMPLE_RATE: f64 = 22050.0;

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

    phase: f64,
}

impl DspNode for SineOscillator {
    fn tick(&mut self) {
        let frequency = (self.input_frequency * 1000.0).abs();
        let phase_diff = (2.0 * std::f64::consts::PI * frequency) / SAMPLE_RATE;
        self.output_output = (self.phase + self.input_modulation).sin();
        self.phase += phase_diff;

        while self.phase > std::f64::consts::PI * 2.0 {
            self.phase -= std::f64::consts::PI * 2.0
        }
    }
}

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

    fn play(&mut self, num_samples: usize) -> Array1<f64> {
        let mut result = Array1::ones(num_samples);

        for element in result.iter_mut() {
            *element = self.tick();
        }

        result
    }
}

#[pymethods]
impl DspGraph {
    #[new]
    fn new() -> Self {
        let mut graph = DspGraph::default();

        let osc = SineOscillator {
            input_frequency: 0.440,
            input_modulation: 0.0,
            output_output: 0.0,
            phase: 0.0,
        };

        let osc_id = graph.add_node(Box::new(osc));

        graph.patch(
            osc_id,
            "output_output",
            graph.speaker_node_id,
            "input_input",
        );

        graph
    }

    fn patch(
        &mut self,
        from_node_id: NodeId,
        from_output_name: &str,
        to_node_id: NodeId,
        to_input_name: &str,
    ) {
        let from_node = self.get_node(from_node_id);
        let to_node = self.get_node(to_node_id);

        let from_output = from_node
            .get_index_of_output(&from_output_name)
            .unwrap_or_else(|| panic!("Output {} not found", from_output_name));

        let to_input = to_node
            .get_index_of_input(&to_input_name)
            .unwrap_or_else(|| panic!("Input {} not found", to_input_name));

        self.connections.push(DspConnection {
            from_node: from_node_id,
            from_output,
            to_node: to_node_id,
            to_input,
        });
    }

    fn tick(&mut self) -> f64 {
        for connection in self.connections.iter() {
            let output_node = &self.nodes[&connection.from_node];
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

    #[pyo3(name = "play")]
    fn play_py<'py>(
        &mut self,
        num_samples: usize,
        py: Python<'py>,
    ) -> &'py PyArray<f64, Dim<[usize; 1]>> {
        self.play(num_samples).into_pyarray(py)
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

    #[test]
    fn test_get_output() {
        let osc = SineOscillator {
            input_frequency: 0.440,
            input_modulation: 0.0,
            output_output: 0.0,
            phase: 0.0,
        };

        osc.get_index_of_output("output_output");
    }

    #[test]
    fn play_basic_graph() {
        let mut g = DspGraph::new();
        g.play(100);
    }

    #[test]
    fn test_node_name() {
        let osc = SineOscillator {
            input_frequency: 0.440,
            input_modulation: 0.0,
            output_output: 0.0,
            phase: 0.0,
        };

        assert_eq!(osc.node_name(), "SineOscillator");
    }
}
