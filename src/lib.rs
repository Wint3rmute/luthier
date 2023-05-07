use numpy::ndarray::{Array1, Dim};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

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

trait DspNode {
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;

    fn input_names(&self) -> &[&str];
    fn output_names(&self) -> &[&str];

    fn set_input(&self, id: InputId, value: f64);
    fn get_output(&self, id: OutputId) -> f64;
}

struct Speaker {}

impl DspNode for Speaker {
    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        0
    }

    fn input_names(&self) -> &[&str] {
        &["input"][..]
    }

    fn output_names(&self) -> &[&str] {
        &[][..]
    }

    fn set_input(&self, id: InputId, value: f64) {}

    fn get_output(&self, id: OutputId) -> f64 {
        0.0
    }
}

#[pyclass]
struct DspGraph {
    nodes: Vec<Box<dyn DspNode + Send>>,
    connections: Vec<DspConnection>,
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

    fn tick(&self) -> f64 {
        0.0
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

impl Default for DspGraph {
    fn default() -> Self {
        Self {
            nodes: vec![],
            connections: vec![],
        }
    }
}

#[pymodule]
fn luthier(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DspGraph>()?;

    Ok(())
}
