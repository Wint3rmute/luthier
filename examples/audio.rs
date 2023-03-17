use luthier::constants::SAMPLE_RATE;
use luthier::node::AudioNodeGraph;
use luthier::node::NodeConnection;
use rodio::Source;

struct SoundEngine {
    graph: AudioNodeGraph,
}

impl Source for SoundEngine {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}

impl Iterator for SoundEngine {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        Some(0.0)
    }
}

fn main() {}
