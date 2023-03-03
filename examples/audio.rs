const SAMPLE_RATE: u32 = 48000;

trait AudioNode {
    fn process(&mut self);

    fn send_to_input(&mut self, input_num: usize, input_value: f32);
    fn get_output(&mut self, output_num: usize) -> f32;
}

struct SineOscillator {
    phase: f32,
    phase_diff: f32,

    phase_mod: f32,
}

impl SineOscillator {
    fn from_frequency(frequency_hz: f32) -> Self {
        Self {
            phase: 0.0,
            phase_mod: 0.0,
            phase_diff: (2.0 * std::f32::consts::PI * frequency_hz) / SAMPLE_RATE as f32,
        }
    }
}

impl AudioNode for SineOscillator {
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

fn main() {}
