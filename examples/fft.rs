use rustfft::{num_complex::Complex, FftPlanner};

fn main() {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(1234);

    let mut buffer = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        1234
    ];

    println!("{:?}", buffer);

    fft.process(&mut buffer);
    std::thread::sleep(std::time::Duration::from_secs(3));
    println!("{:?}", buffer);
}
