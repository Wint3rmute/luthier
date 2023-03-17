use eframe::{
    egui::{self, Painter},
    epaint::Color32,
};
use luthier::node::SineOscillator;
use luthier::node::{AudioNodeGraph, Constant};

fn main() -> Result<(), eframe::Error> {
    // Log to stdout (if you run with `RUST_LOG=debug`).
    // tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(320.0, 240.0)),
        ..Default::default()
    };

    let mut app = MyApp::default();
    app.graph.nodes.push(Box::new(SineOscillator::default()));
    app.graph.nodes.push(Box::new(Constant::default()));

    eframe::run_native("My egui App", options, Box::new(|_cc| Box::new(app)))
}

struct MyApp {
    pub graph: AudioNodeGraph,
    // node_position_info:
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            graph: AudioNodeGraph::default(),
        }
    }
}

const DEFAULT_STROKE: egui::Stroke = egui::Stroke {
    width: 1.0,
    color: Color32::LIGHT_RED,
};

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        for node in self.graph.nodes.iter() {
            egui::Window::new(node.name()).show(ctx, |ui| {
                // ui.horizontal(add_contents);
                // ui.button("Label 1");

                ui.horizontal_wrapped(|ui| {
                    // ui.spacing_mut().item_spacing.x = 0.0; // remove spacing between widgets
                    // `radio_value` also works for enums, integers, and more.
                    ui.vertical(|ui| {
                        for input in node.inputs().iter() {
                            ui.button(*input);
                        }
                    });

                    // TODO: check if a RTL layout can be used for better looks
                    // ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                    //     for output in node.outputs().iter() {
                    //         ui.button(*output);
                    //     }
                    // })

                    ui.vertical(|ui| {
                        for output in node.outputs().iter() {
                            ui.button(*output);
                        }
                    });
                });
                // ui.label("Label 2");
            });
        }
        // let mut win_1_position: Option<egui::Response> = None;
        // let mut win_2_position: Option<egui::Response> = None;

        // egui::Window::new("Node1").show(ctx, |ui| {
        //     win_1_position = Some(ui.label("Label 1"));
        // });

        // egui::Window::new("Node2").show(ctx, |ui| {
        //     win_2_position = Some(ui.label("Label 2"));
        // });

        // egui::CentralPanel::default().show(ctx, |ui| {
        //     if let (Some(w1_position), Some(w2_position)) = (win_1_position, win_2_position) {
        //         let painter = Painter::new(
        //             ui.ctx().clone(),
        //             ui.layer_id(),
        //             ui.available_rect_before_wrap(),
        //         );

        //         painter.add(egui::Shape::line_segment(
        //             [w1_position.rect.min, w2_position.rect.min],
        //             egui::Stroke {
        //                 width: 3.0,
        //                 color: Color32::LIGHT_RED,
        //             },
        //         ));
        //     }
        // });
    }
}
