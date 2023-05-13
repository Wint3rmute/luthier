use luthier::DspGraph;

use rodio::{OutputStream, Sink};

fn main() {
    let mut g = DspGraph::new();

    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    sink.append(g);

    println!("Playing!");
    sink.sleep_until_end();
}

