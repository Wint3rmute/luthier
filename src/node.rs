use crate::constants::SAMPLE_RATE;

pub trait AudioNode {
    fn process(&mut self);

    fn send_to_input(&mut self, input_num: usize, input_value: f32);
    fn get_output(&mut self, output_num: usize) -> f32;

    fn name(&self) -> &str;

    fn inputs(&self) -> &[&str];
    fn outputs(&self) -> &[&str];
}

pub struct SineOscillator {
    pub phase: f32,
    pub phase_diff: f32,
    pub phase_mod: f32,
}

impl Default for SineOscillator {
    fn default() -> Self {
        Self {
            phase: 0.0,
            phase_diff: 0.0,
            phase_mod: 0.0,
        }
    }
}

pub struct Constant {
    pub value: f32,
}

impl Default for Constant {
    fn default() -> Self {
        Self { value: 0.0 }
    }
}

impl AudioNode for Constant {
    fn name(&self) -> &str {
        "Constant"
    }

    fn inputs(&self) -> &[&str] {
        [].as_slice()
    }

    fn outputs(&self) -> &[&str] {
        ["value"].as_slice()
    }

    fn process(&mut self) {}

    fn get_output(&mut self, output_num: usize) -> f32 {
        self.value
    }

    fn send_to_input(&mut self, input_num: usize, input_value: f32) {}
}

impl SineOscillator {
    pub fn from_frequency(frequency_hz: f32) -> Self {
        Self {
            phase: 0.0,
            phase_mod: 0.0,
            phase_diff: (2.0 * std::f32::consts::PI * frequency_hz) / SAMPLE_RATE as f32,
        }
    }
}

impl AudioNode for SineOscillator {
    fn inputs(&self) -> &[&str] {
        ["freq", "phase_mod"].as_slice()
    }

    fn outputs(&self) -> &[&str] {
        ["signal"].as_slice()
    }

    fn name(&self) -> &str {
        return "SineOscillator";
    }

    fn process(&mut self) {}

    fn get_output(&mut self, output_num: usize) -> f32 {
        self.phase += self.phase_diff;
        let result = (self.phase + self.phase_mod).sin();

        result
    }

    fn send_to_input(&mut self, input_num: usize, input_value: f32) {
        self.phase_mod = input_value;
    }
}

pub struct NodeConnection {
    from_node: usize,
    to_node: usize,
    from_node_slot_id: usize,
    to_node_slot_id: usize,
}

pub struct AudioNodeGraph {
    pub nodes: Vec<Box<dyn AudioNode + Send>>,
    pub connections: Vec<NodeConnection>,
}

impl Default for AudioNodeGraph {
    fn default() -> Self {
        Self {
            nodes: vec![],
            connections: vec![],
        }
    }
}
