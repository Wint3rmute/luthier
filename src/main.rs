pub mod constants;
pub mod node;
use node::SineOscillator;

fn main() {
    let n = SineOscillator {
        phase: 0.0,
        phase_diff: 0.0,
        phase_mod: 0.0,
    };

    println!("Hello, world!");
}
