use pyo3::exceptions::PyValueError;
use rand::prelude::*;
use rodio::source::Source;
use std::collections::BTreeMap;
use std::io::Write;
use std::time::Duration;

pub mod ladder_filter;
pub mod mverb;
use ladder_filter::LadderFilter;

use node_traits::{DspConnectible, DspNode, InputId, Node, NodeId, OutputId};
use numpy::ndarray::{Array1, Dim};
use numpy::{IntoPyArray, PyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::process::{Command, Stdio};

extern crate node_macro;
use node_macro::DspConnectibleDerive;

// const SAMPLE_RATE: f64 = 22050.0;
const SAMPLE_RATE: f64 = 48000.0;

#[derive(Debug)]
struct DspConnection {
    from_node: NodeId,
    from_output: OutputId,
    to_node: NodeId,
    to_input: InputId,
}

#[pyclass(freelist = 64)]
#[derive(Default, Clone, DspConnectibleDerive)]
struct Speaker {
    input_input: f64,
}

impl DspNode for Speaker {
    fn tick(&mut self) {}
}

#[derive(DspConnectibleDerive, Clone)]
struct BaseFrequency {
    input_base_frequency: f64,
    output_output: f64,
}

impl Default for BaseFrequency {
    fn default() -> Self {
        BaseFrequency {
            input_base_frequency: 0.440,
            output_output: 0.440,
        }
    }
}

impl DspNode for BaseFrequency {
    fn tick(&mut self) {
        self.output_output = self.input_base_frequency
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
pub struct Chorus {
    input_input: f64,
    input_feedback: f64,
    input_dry: f64,
    input_wet: f64,

    output_output: f64,

    buffer_length_lfo_state: f64,
    buffer_index: usize,
    buffer: [f64; 1000],
}

impl Default for Chorus {
    fn default() -> Self {
        Self {
            input_input: 0.0,
            input_feedback: 0.0,
            input_dry: 1.0,
            input_wet: 0.4,

            output_output: 0.0,

            buffer_length_lfo_state: 0.0,

            buffer_index: 0,
            buffer: [0.0; 1000],
        }
    }
}

#[pymethods]
impl Chorus {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for Chorus {
    fn tick(&mut self) {
        self.output_output =
            self.input_input * self.input_dry + self.buffer[self.buffer_index] * self.input_wet;

        self.buffer[self.buffer_index] =
            self.buffer[self.buffer_index] * self.input_feedback + self.input_input;

        self.buffer_index += 1;
        self.buffer_length_lfo_state += 1.0 / SAMPLE_RATE;
        self.buffer_length_lfo_state %= 2.0 * std::f64::consts::PI;

        if self.buffer_index >= self.buffer.len() - (self.buffer_length_lfo_state * 100.0) as usize
        {
            self.buffer_index = 0;
        }
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
pub struct Delay {
    input_input: f64,
    input_length: f64,
    input_feedback: f64,
    input_dry: f64,
    input_wet: f64,

    output_output: f64,

    buffer: [f64; SAMPLE_RATE as usize * 2],
    buffer_index: usize,
}

impl Default for Delay {
    fn default() -> Self {
        Self {
            input_input: 0.0,
            input_length: 0.3,
            input_feedback: 0.3,
            input_dry: 0.9,
            input_wet: 0.3,
            output_output: 0.0,
            buffer: [0.0; SAMPLE_RATE as usize * 2],
            buffer_index: 0,
        }
    }
}

#[pymethods]
impl Delay {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for Delay {
    fn tick(&mut self) {
        self.output_output =
            self.input_input * self.input_dry + self.buffer[self.buffer_index] * self.input_wet;

        self.buffer[self.buffer_index] =
            self.buffer[self.buffer_index] * self.input_feedback + self.input_input;

        self.buffer_index += 1;
        if self.buffer_index
            >= (self.buffer.len() as f64 * (self.input_length + 1.0) / 2.0) as usize
        {
            self.buffer_index = 0;
        }
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
pub struct Reverb {
    input_input: f64,
    input_size: f64,
    input_decay: f64,
    input_dry: f64,
    input_wet: f64,

    output_output: f64,

    mverb: mverb::MVerb,
}

impl Default for Reverb {
    fn default() -> Self {
        Self {
            input_input: 0.0,
            input_size: 0.5,
            input_decay: 0.5,
            input_dry: 0.5,
            input_wet: 0.5,

            output_output: 0.0,
            mverb: mverb::MVerb::default(),
        }
    }
}

#[pymethods]
impl Reverb {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for Reverb {
    fn tick(&mut self) {
        // TODO: add dry/wet and reverb params as inputs
        self.mverb.size = self.input_size;
        self.mverb.decay = self.input_decay;

        let wet = self.mverb.process((self.input_input, self.input_input)).0;
        let dry = self.input_input;
        self.output_output = dry * self.input_dry + wet * self.input_wet;
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
pub struct HighPassFilter {
    input_cutoff: f64,
    input_input: f64,
    output_output: f64,
    prev_input: f64,
    prev_output: f64,
}

#[pymethods]
impl HighPassFilter {
    #[new]
    fn new() -> Self {
        Self {
            input_cutoff: 0.5,
            input_input: 0.0,
            output_output: 0.0,
            // filter: LadderFilter::default(),
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }
}

impl DspNode for HighPassFilter {
    fn tick(&mut self) {
        let dt = 1.0 / SAMPLE_RATE;
        let rc =
            1.0 / (2.0 * std::f64::consts::PI * (1.0 - self.input_cutoff.abs() + 0.0001) * 20000.);
        let alpha = rc / (rc + dt);

        let output = alpha * (self.prev_output + self.input_input - self.prev_input);
        self.prev_input = self.input_input;
        self.prev_output = output;

        self.output_output = output;
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
pub struct LowPassFilter {
    input_cutoff: f64,
    input_resonance: f64,
    input_input: f64,
    output_output: f64,
    filter: LadderFilter,
}

#[pymethods]
impl LowPassFilter {
    #[new]
    fn new() -> Self {
        Self {
            input_cutoff: 0.5,
            input_resonance: 0.0,
            input_input: 0.0,
            output_output: 0.0,
            filter: LadderFilter::default(),
        }
    }
}

impl DspNode for LowPassFilter {
    fn tick(&mut self) {
        self.filter.params.res = self.input_resonance;
        self.filter.params.set_cutoff(self.input_cutoff.abs());
        self.output_output = self.filter.process(self.input_input);
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone, Default)]
pub struct TriangleOscillator {
    input_frequency: f64,
    input_detune: f64,
    // input_pwm: f64,
    output_output: f64,
    phase: f64,
    going_up: bool,
}

#[pymethods]
impl TriangleOscillator {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for TriangleOscillator {
    fn tick(&mut self) {
        let frequency =
            (self.input_frequency * 2000.0).abs() + self.input_detune * self.input_frequency * 20.0;
        let phase_diff = frequency * 2.0 / SAMPLE_RATE;

        self.output_output = self.phase;

        if self.going_up {
            self.phase += phase_diff;
        } else {
            self.phase -= phase_diff;
        }

        if self.phase > 1.0 {
            self.going_up = false;
        }
        if self.phase < -1.0 {
            self.going_up = true;
        }
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone, Default)]
pub struct SawOscillator {
    input_frequency: f64,
    input_detune: f64,
    input_pwm: f64,
    output_output: f64,
    phase: f64,
}

#[pymethods]
impl SawOscillator {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for SawOscillator {
    fn tick(&mut self) {
        let frequency =
            (self.input_frequency * 1000.0).abs() + self.input_detune * self.input_frequency * 20.0;
        let phase_diff = frequency * 2.0 / SAMPLE_RATE;

        self.output_output = self.phase;

        self.phase += phase_diff;

        while self.phase > 1.0 {
            self.phase = -1.0
        }
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone, Default)]
pub struct SquareOscillator {
    input_frequency: f64,
    input_pwm: f64,
    input_detune: f64,
    output_output: f64,
    phase: f64,
}

#[pymethods]
impl SquareOscillator {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for SquareOscillator {
    fn tick(&mut self) {
        let frequency =
            (self.input_frequency * 1000.0).abs() + self.input_detune * self.input_frequency * 20.0;
        let phase_diff = frequency * 2.0 / SAMPLE_RATE;

        self.output_output = if (self.phase - self.input_pwm) > 0.0 {
            1.0
        } else {
            -1.0
        };

        self.phase += phase_diff;

        while self.phase > 1.0 {
            self.phase = -1.0;
        }
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
struct KarplusStrong {
    input_frequency: f64,
    input_feedback: f64,
    input_invert: f64,

    output_output: f64,

    buffer: [f64; (SAMPLE_RATE * 2.0) as usize],
    buffer_index: usize,
}

#[pymethods]
impl KarplusStrong {
    #[new]
    fn new() -> Self {
        let mut result = Self {
            input_frequency: 0.220,
            input_feedback: 0.0,
            input_invert: 0.0,
            output_output: 0.0,
            buffer: [0.0; (SAMPLE_RATE * 2.0) as usize],
            buffer_index: 0,
        };

        let mut rng = rand::thread_rng();
        for sample in result.buffer.iter_mut() {
            *sample = rng.gen::<f64>() - 0.5;
        }

        result
    }
}

impl DspNode for KarplusStrong {
    fn tick(&mut self) {
        let previous_value = self.buffer[self.buffer_index];

        self.buffer_index += 1;
        let mut frequency = (self.input_frequency * 1000.0).abs();
        if frequency < 50. {
            frequency = 50.0
        }

        let buffer_max = (100000.0 / frequency) as usize;
        if self.buffer_index >= buffer_max {
            self.buffer_index = 0;
        }

        let feedback = (self.input_feedback + 1.0) / 2.0;
        let one_minus_feedback = 1.0 - feedback;

        self.output_output = self.buffer[self.buffer_index];
        self.buffer[self.buffer_index] =
            self.buffer[self.buffer_index] * feedback + previous_value * one_minus_feedback;
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
struct SineOscillator {
    input_frequency: f64,
    input_modulation: f64,
    input_modulation_index: f64,
    output_output: f64,

    phase: f64,
}

#[pymethods]
impl SineOscillator {
    #[new]
    fn new() -> Self {
        return SineOscillator {
            input_frequency: 0.0,
            input_modulation: 0.0,
            input_modulation_index: 0.1,
            output_output: 0.0,
            phase: 0.0,
        };
    }
}

impl DspNode for SineOscillator {
    fn tick(&mut self) {
        let frequency = (self.input_frequency * 1000.0).abs();
        let phase_diff = (2.0 * std::f64::consts::PI * frequency) / SAMPLE_RATE;
        self.output_output =
            (self.phase + self.input_modulation * self.input_modulation_index * 100.0).sin();
        self.phase += phase_diff;

        while self.phase > std::f64::consts::PI * 2.0 {
            self.phase -= std::f64::consts::PI * 2.0
        }
    }
}

#[pyclass]
#[derive(PartialEq, Eq, Clone)]
enum AdsrPhase {
    ATTACK,
    SUSTAIN,
    RELEASE,
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(DspConnectibleDerive, Clone)]
struct ADSR {
    input_attack: f64,
    input_sustain: f64,
    input_release: f64,

    output_output: f64,

    phase: AdsrPhase,
    state: f64,
    sustain_state: f64,
}

#[pymethods]
impl ADSR {
    #[new]
    fn new() -> Self {
        Self {
            input_attack: 0.1,
            input_sustain: 0.1,
            input_release: 0.1,
            output_output: 0.0,

            phase: AdsrPhase::ATTACK,
            state: 0.0,
            sustain_state: 0.0,
        }
    }
}

impl DspNode for ADSR {
    fn tick(&mut self) {
        if self.phase == AdsrPhase::ATTACK {
            let attack = (self.input_attack + 1.0).abs() / 2.0;
            let state_inc = 0.4 / attack / SAMPLE_RATE;
            self.state += state_inc;
            if self.state > 1.0 {
                self.state = 1.0;
                self.phase = AdsrPhase::SUSTAIN;
            }
        } else if self.phase == AdsrPhase::SUSTAIN {
            let sustain = (self.input_sustain + 1.0).abs() / 2.0;
            let state_inc = 0.4 / sustain / SAMPLE_RATE;
            self.sustain_state += state_inc;
            if self.sustain_state >= 1.0 {
                self.phase = AdsrPhase::RELEASE;
            }
        } else if self.phase == AdsrPhase::RELEASE {
            let release = (self.input_release + 1.0).abs() / 2.0;
            let state_dec = 0.4 / release / SAMPLE_RATE;
            self.state -= state_dec;
            if self.state < 0.0 {
                self.state = 0.0
            }
        }

        self.output_output = self.state;
    }
}

const HARMONICS: [f64; 12] = [0.1, 0.15, 0.2, 0.25, 0.4, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0];

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(Clone, Default, DspConnectibleDerive)]
struct HarmonicMultiplier {
    input_input: f64,
    input_scale: f64,
    output_output: f64,
}

#[pymethods]
impl HarmonicMultiplier {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for HarmonicMultiplier {
    fn tick(&mut self) {
        self.output_output = self.input_input
            * HARMONICS
                [(((self.input_scale + 1.0) / 2.0) * (HARMONICS.len() as f64) - 1.0) as usize];
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(Clone, Default, DspConnectibleDerive)]
struct Sum {
    input_in_1: f64,
    input_in_2: f64,
    input_in_3: f64,
    input_in_4: f64,
    output_output: f64,
}

impl DspNode for Sum {
    fn tick(&mut self) {
        self.output_output =
            (self.input_in_1 + self.input_in_2 + self.input_in_3 + self.input_in_4) / 4.0;
    }
}

#[pymethods]
impl Sum {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

#[pyclass(set_all, get_all, freelist = 64)]
#[derive(Clone, DspConnectibleDerive)]
struct Multiplier {
    input_input: f64,
    input_scale: f64,
    output_output: f64,
}

impl Default for Multiplier {
    fn default() -> Self {
        Self {
            input_input: 0.0,
            input_scale: 0.5,
            output_output: 0.0,
        }
    }
}

#[pymethods]
impl Multiplier {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

impl DspNode for Multiplier {
    fn tick(&mut self) {
        self.output_output = self.input_input * self.input_scale;
    }
}

#[pyclass(freelist = 64)]
pub struct DspGraph {
    nodes: BTreeMap<NodeId, Box<dyn DspNode>>,
    connections: Vec<DspConnection>,
    current_node_index: NodeId,

    #[pyo3(get)]
    pub speaker_node_id: NodeId,
    #[pyo3(get)]
    pub base_frequency_node_id: NodeId,
}

impl Default for DspGraph {
    fn default() -> Self {
        let mut result = Self {
            nodes: BTreeMap::new(),
            connections: vec![],
            current_node_index: 0,

            speaker_node_id: 0,
            base_frequency_node_id: 0,
        };

        result.speaker_node_id = result.add_node(Box::new(Speaker::default()));
        result.base_frequency_node_id = result.add_node(Box::new(BaseFrequency::default()));

        result
    }
}

impl Iterator for DspGraph {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.tick() as f32)
    }
}

impl Source for DspGraph {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

impl DspGraph {
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let node_index = self.get_next_node_index() - 1; // patola
        self.nodes.insert(node_index, node);
        node_index
    }

    fn get_next_node_index(&mut self) -> usize {
        self.current_node_index += 1;
        self.current_node_index
    }

    fn get_node(&self, node_id: NodeId) -> &Node {
        self.nodes
            .get(&node_id)
            .unwrap_or_else(|| panic!("Node with id {} not found", node_id))
    }

    fn get_node_mut(&mut self, node_id: NodeId) -> &mut Node {
        self.nodes
            .get_mut(&node_id)
            .unwrap_or_else(|| panic!("Node with id {} not found", node_id))
    }

    fn play(&mut self, num_samples: usize) -> Array1<f64> {
        let mut result = Array1::ones(num_samples);

        for element in result.iter_mut() {
            *element = self.tick();
        }

        result
    }

    fn draw(&self) -> Vec<u8> {
        let graphviz_code = self.get_graphviz_code();

        let mut graphviz_process = Command::new("dot")
            .arg("-T")
            .arg("png")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Unable to start graphviz");

        let mut graphviz_stdin = graphviz_process
            .stdin
            .take()
            .expect("Unable to connect to graphviz process stdin");
        std::thread::spawn(move || {
            graphviz_stdin
                .write_all(graphviz_code.as_bytes())
                .expect("Unable to write data to graphviz stdin");
        });

        let output = graphviz_process
            .wait_with_output()
            .expect("failed to wait on graphviz output");

        output.stdout //.clone()
    }
}

impl DspGraph {
    fn inputs_iterator(&self) -> impl Iterator<Item = (NodeId, InputId)> + '_ {
        self.nodes.iter().flat_map(|(node_id, node)| {
            node.get_input_names()
                .iter()
                .enumerate()
                .filter_map(|(input_id, input_name)| {
                    if !self.is_modulated(*node_id, input_id)
                        // Awful hack to avoid setting mixer input values :)
                        && *input_name != "input_base_frequency"
                        && *input_name != "input_in_1"
                        && *input_name != "input_in_2"
                        && *input_name != "input_in_3"
                        && *input_name != "input_in_4"
                    {
                        Some((*node_id, input_id))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
    }
}

#[pymethods]
impl DspGraph {
    #[new]
    pub fn new() -> Self {
        DspGraph::default()
    }

    fn num_inputs(&self) -> usize {
        self.inputs_iterator().count()
    }

    fn set_inputs(&mut self, inputs: &PyArrayDyn<f64>) -> PyResult<()> {
        let inputs = unsafe { inputs.as_array() };

        let graph_inputs = self.num_inputs();
        if !inputs.len() == graph_inputs {
            return Err(PyValueError::new_err(format!("Input array's length must be the same as the number of inputs in the DspGraph: {graph_inputs}")));
        }

        let inputs_iterator_collected: Vec<(NodeId, InputId)> = self.inputs_iterator().collect();

        for ((node_id, input_id), new_value) in inputs_iterator_collected.iter().zip(inputs) {
            self.get_node_mut(*node_id)
                .set_input_by_index(*input_id, *new_value);
        }

        Ok(())
    }

    fn get_inputs<'py>(&mut self, py: Python<'py>) -> &'py PyArray<f64, Dim<[usize; 1]>> {
        self.inputs_iterator()
            .map(|(node_id, input_id)| self.get_node(node_id).get_input_by_index(input_id))
            .collect::<Array1<f64>>()
            .into_pyarray(py)
    }

    fn is_modulated(&self, node_id: NodeId, input_id: InputId) -> bool {
        for connection in self.connections.iter() {
            if connection.to_node == node_id && connection.to_input == input_id {
                return true;
            }
        }

        false
    }

    fn add_triangle(&mut self, triangle: TriangleOscillator) -> NodeId {
        self.add_node(Box::new(triangle))
    }

    fn add_square(&mut self, square: SquareOscillator) -> NodeId {
        self.add_node(Box::new(square))
    }

    fn add_sine(&mut self, sine: SineOscillator) -> NodeId {
        self.add_node(Box::new(sine))
    }

    fn add_karplus_strong(&mut self, karplus_strong: KarplusStrong) -> NodeId {
        self.add_node(Box::new(karplus_strong))
    }

    fn add_saw(&mut self, saw: SawOscillator) -> NodeId {
        self.add_node(Box::new(saw))
    }

    fn add_lowpass(&mut self, lowpass: LowPassFilter) -> NodeId {
        self.add_node(Box::new(lowpass))
    }

    fn add_highpass(&mut self, highpass: HighPassFilter) -> NodeId {
        self.add_node(Box::new(highpass))
    }

    fn add_sum(&mut self, sum: Sum) -> NodeId {
        self.add_node(Box::new(sum))
    }

    fn add_adsr(&mut self, adsr: ADSR) -> NodeId {
        self.add_node(Box::new(adsr))
    }

    fn add_delay(&mut self, delay: Delay) -> NodeId {
        self.add_node(Box::new(delay))
    }

    fn add_chorus(&mut self, chorus: Chorus) -> NodeId {
        self.add_node(Box::new(chorus))
    }

    fn add_reverb(&mut self, reverb: Reverb) -> NodeId {
        self.add_node(Box::new(reverb))
    }

    fn add_multiplier(&mut self, multiplier: Multiplier) -> NodeId {
        self.add_node(Box::new(multiplier))
    }

    fn add_harmonic_multiplier(&mut self, multiplier: HarmonicMultiplier) -> NodeId {
        self.add_node(Box::new(multiplier))
    }

    fn randomize_inputs<'py>(&mut self) {
        let mut rng = rand::thread_rng();
        let inputs_iterator_collected: Vec<(NodeId, InputId)> = self.inputs_iterator().collect();

        for (node_id, input_id) in inputs_iterator_collected.iter() {
            let new_value = rng.gen_range(-1.0..1.0);
            self.get_node_mut(*node_id)
                .set_input_by_index(*input_id, new_value);
        }
    }

    fn get_graphviz_code(&self) -> String {
        let mut graphviz_code = String::new();
        let gradient = colorous::RED_BLUE;

        let node_to_color = std::collections::HashMap::from([
            ("BaseFrequency", "#82AAFF"),
            // Sources
            ("SineOscillator", "#FF5370"),
            ("SquareOscillator", "#FF5370"),
            ("TriangleOscillator", "#FF5370"),
            ("SawOscillator", "#FF5370"),
            ("KarplusStrong", "#FF5370"),
            //Filters
            ("HighPassFilter", "#FF5370"),
            ("LowPassFilter", "#FF5370"),
            //Effects
            ("Delay", "#C3E88D"),
            ("Reverb", "#C3E88D"),
            ("Chorus", "#C3E88D"),
            // Control signals
            ("ADSR", "#FFCB6B"),
            ("Sum", "#FFCB6B"),
            ("Multiplier", "#FFCB6B"),
            ("HarmonicMultiplier", "#FFCB6B"),
        ]);

        graphviz_code.push_str(
            r#"digraph g {
splines="polyline"
rankdir = "LR"

fontname="Fira Code"
node [fontname="Fira Code"]
"#,
        );
        for (node_id, node) in self.nodes.iter() {
            let node_color = node_to_color
                .get(node.node_name())
                .unwrap_or_else(|| &"white");

            let node_name = node.node_name();
            graphviz_code.push_str(
                format!(
                    r#"
"node{node_id}" [
    shape = none
    label = <<table border="0" cellspacing="0">
    <tr><td border="1" bgcolor="{node_color}">{node_name} #{node_id}</td></tr>
            "#
                )
                .as_str(),
            );

            for (input_id, input) in node.get_input_names().iter().enumerate() {
                let input_value = node.get_input_by_index(input_id);
                let color = gradient.eval_continuous((input_value + 1.0) / 2.0);
                let color = format!("#{:x}", color);

                let input_icon = if self.is_modulated(*node_id, input_id) {
                    "●"
                } else {
                    "○"
                };

                graphviz_code.push_str(
                    format!(r#"<tr><td border="1" bgcolor="{color}" port="{input}"> {input_icon} {input}: {input_value:.3} </td></tr> \n"#)
                        .as_str(),
                );
            }

            for output in node.get_output_names() {
                graphviz_code.push_str(
                    format!(
                        r#"<tr><td border="1" port="{output}"><b> {output} ● </b></td></tr> \n"#
                    )
                    .as_str(),
                );
            }

            graphviz_code.push_str("</table>>\n];");
        }

        for connection in &self.connections {
            let output_name =
                &self.nodes[&connection.from_node].get_output_names()[connection.from_output];
            let input_name =
                &self.nodes[&connection.to_node].get_input_names()[connection.to_input];

            let from_node = &connection.from_node;
            let to_node = &connection.to_node;

            graphviz_code.push_str(
                format!(
                    r#"
                "node{from_node}":{output_name} -> "node{to_node}":{input_name} [];
                "#
                )
                .as_str(),
            );
        }

        graphviz_code.push_str("\n}");
        graphviz_code
    }

    #[pyo3(name = "draw")]
    fn draw_py<'py>(&mut self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.draw())
    }

    fn set_input(&mut self, node_id: NodeId, input_name: &str, value: f64) {
        let node = self.get_node_mut(node_id);
        let node_name = node.node_name();
        let input_index = node
            .get_index_of_input(input_name)
            .unwrap_or_else(|| panic!("Input {input_name} not found in {node_name}"));
        // .expect(format!("Input {input_name} not found in {node_name}")
        node.set_input_by_index(input_index, value);
    }

    pub fn patch(
        &mut self,
        from_node_id: NodeId,
        from_output_name: &str,
        to_node_id: NodeId,
        to_input_name: &str,
    ) {
        let from_node = self.get_node(from_node_id);
        let to_node = self.get_node(to_node_id);

        let from_output = from_node
            .get_index_of_output(from_output_name)
            .unwrap_or_else(|| panic!("Output {} not found", from_output_name));

        let to_input = to_node
            .get_index_of_input(to_input_name)
            .unwrap_or_else(|| panic!("Input {} not found", to_input_name));

        self.connections.push(DspConnection {
            from_node: from_node_id,
            from_output,
            to_node: to_node_id,
            to_input,
        });
    }

    pub fn tick(&mut self) -> f64 {
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
    m.add_class::<KarplusStrong>()?;
    m.add_class::<Reverb>()?;
    m.add_class::<Sum>()?;
    m.add_class::<SineOscillator>()?;
    m.add_class::<SquareOscillator>()?;
    m.add_class::<TriangleOscillator>()?;
    m.add_class::<SawOscillator>()?;
    m.add_class::<ADSR>()?;
    m.add_class::<Multiplier>()?;
    m.add_class::<HarmonicMultiplier>()?;
    m.add_class::<LowPassFilter>()?;
    m.add_class::<HighPassFilter>()?;
    m.add_class::<Delay>()?;
    m.add_class::<Chorus>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_output() {
        let osc = SawOscillator {
            input_frequency: 0.440,
            input_detune: 0.0,
            output_output: 0.0,
            input_pwm: 0.0,
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
    fn test_draw_doesnt_panic() {
        let g = DspGraph::new();
        g.get_graphviz_code();
        g.draw();
    }

    #[test]
    fn test_randomise_inputs_doesnt_panic() {
        let mut g = DspGraph::new();
        g.randomize_inputs();
    }

    #[test]
    fn test_mixer_inputs_dont_count_as_twekable_inputs() {
        let mut g = DspGraph::new();
        let inputs_before = g.num_inputs();

        g.add_node(Box::new(Sum::new()));
        assert_eq!(g.num_inputs(), inputs_before);
    }

    #[test]
    fn test_node_name() {
        let osc = SineOscillator {
            input_frequency: 0.440,
            input_modulation_index: 0.1,
            input_modulation: 0.0,
            output_output: 0.0,
            phase: 0.0,
        };

        assert_eq!(osc.node_name(), "SineOscillator");
    }

    #[test]
    fn test_set_field_by_proc_macro_enum() {
        let mut osc = SineOscillator {
            input_frequency: 0.440,
            input_modulation_index: 0.1,
            input_modulation: 0.0,
            output_output: 0.0,
            phase: 0.0,
        };

        osc.set_input_by_index(SineOscillatorInputs::INPUT_FREQUENCY as usize, 1.0);
        assert_eq!(osc.input_frequency, 1.0);
    }
}
