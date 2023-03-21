use luthier::constants::SAMPLE_RATE;
use luthier::node::AudioNode;
use luthier::node::AudioNodeGraph;
use luthier::node::NodeConnection;
use luthier::node::SineOscillator;
use rodio::OutputStream;
use rodio::Sink;
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
        // let mut result = 0.0;

        Some(
            self.graph
                .nodes
                .iter_mut()
                .map(|node| {
                    node.process();
                    node.get_output(0)
                })
                .sum(),
        )
    }
}

fn main() {
    // Get a output stream handle to the default physical sound device
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    let engine = SoundEngine {
        graph: AudioNodeGraph {
            nodes: vec![Box::new(SineOscillator::from_frequency(440.))],
            connections: vec![],
        },
    };

    sink.append(engine);
    sink.sleep_until_end();
}
