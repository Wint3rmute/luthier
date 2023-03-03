use eframe::{
    egui::{self, Painter},
    epaint::Color32,
};

fn main() -> Result<(), eframe::Error> {
    // Log to stdout (if you run with `RUST_LOG=debug`).
    // tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(320.0, 240.0)),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    )
}

struct MyApp;

impl Default for MyApp {
    fn default() -> Self {
        Self {}
    }
}

const DEFAULT_STROKE: egui::Stroke = egui::Stroke {
    width: 1.0,
    color: Color32::LIGHT_RED,
};

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut win_1_position: Option<egui::Response> = None;
        let mut win_2_position: Option<egui::Response> = None;

        egui::Window::new("Node1").show(ctx, |ui| {
            win_1_position = Some(ui.label("Label 1"));
        });

        egui::Window::new("Node2").show(ctx, |ui| {
            win_2_position = Some(ui.label("Label 2"));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let (Some(w1_position), Some(w2_position)) = (win_1_position, win_2_position) {
                let painter = Painter::new(
                    ui.ctx().clone(),
                    ui.layer_id(),
                    ui.available_rect_before_wrap(),
                );
                painter.add(egui::Shape::line_segment(
                    [w1_position.rect.min, w2_position.rect.min],
                    DEFAULT_STROKE,
                ));
            }
        });
    }
}
