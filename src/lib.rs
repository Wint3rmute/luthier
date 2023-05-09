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

struct Speaker {
    input_input: f64,
}

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

#[pyclass]
#[derive(Default)]
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

#[pymodule]
fn luthier(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DspGraph>()?;

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
