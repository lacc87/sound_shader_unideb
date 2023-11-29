use std::fmt::Display;
use nalgebra::Matrix4;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, FromSample, Sample, SampleFormat};
use gtk4::prelude::*;
use gtk4::{
    Label, Grid, glib, Align, Application, ApplicationWindow, Button, DrawingArea, DropDown, StringList,
};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
mod audio_helper;
use wgpu::{util::DeviceExt, *};

const APP_ID: &str = "hu.unideb.rust_sound_shader";
static AUDIO_DATA: Mutex<Vec<f32>> = Mutex::new(vec![]);
static GRAPH_DATA: Mutex<Vec<f32>> = Mutex::new(vec![]);

const SHADER_PREFIX: &str = "#version 450
layout(local_size_x = 1) in;

layout(set = 0, binding = 0) buffer OutputStorage {
	vec2[] output;
};

layout(set = 0, binding = 1) uniform DeviceInfo {
	uint iSampleRate;
	uint iBaseFrame;
};
";

const SHADER_SUFFIX: &str = "
void main() {
	uint idx = gl_GlobalInvocationID.x;
	uint frame = iBaseFrame + idx;
	output[idx] = mainSound(idx, float(frame) / float(iSampleRate));
}
";

fn main() -> glib::ExitCode {
    let app = Application::builder().application_id(APP_ID).build();

    app.connect_activate(build_ui);

    app.run()
}

fn init_device() -> () {
	let instance = Instance::new(Backends::PRIMARY);
	pollster::block_on(async {
        println!("alkdlaskd léaskdlé asklédas");
		let adaptor = instance
			.request_adapter(&Default::default())
			.await
			.expect("failed to find an appropriate adapter");
		let (device, queue) = adaptor
			.request_device(&Default::default(), None)
			.await
			.expect("failed to create device");


    // Create buffer with initial data
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
    });


    let bind_group_layout = &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
				binding: 0,
				visibility: ShaderStages::COMPUTE,
				ty: BindingType::Buffer {
					ty: BufferBindingType::Storage { read_only: false },
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			},
			BindGroupLayoutEntry {
				binding: 1,
				visibility: ShaderStages::COMPUTE,
				ty: BindingType::Buffer {
					ty: BufferBindingType::Storage { read_only: false },
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}
        ],
    });

    // Create buffer to receive the result
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<f32>() as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Create bind group for the compute shader
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        },wgpu::BindGroupEntry {
            binding: 1,
            resource: result_buffer.as_entire_binding(),
        }],
    });

    // Load and compile the compute shader
    let compute_shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader.wgsl").into())
    });

    // Create the compute pipeline
    let compute_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            //bind_group_layouts: &[&bind_group.get_layout()],
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader_module,
        entry_point: "main", // Entry point in the compute shader
    });

    // Create a command encoder to run the compute shader
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        // Start a compute pass
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
        });

        // Set pipeline and bind group
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch the compute shader with a single workgroup
        compute_pass.dispatch(1, 1, 1);
    }

    // Copy the result from the GPU to the CPU
    encoder.copy_buffer_to_buffer(
        &buffer,
        0,
        &result_buffer,
        0,
        std::mem::size_of::<f32>() as u64 * 4,
    );

    // Submit the command encoder
    queue.submit(std::iter::once(encoder.finish()));

    // Map the result buffer to read the data
    let result_slice = result_buffer.slice(..);
    let result_data = result_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    // Wait for mapping to complete
    result_data.await.unwrap();

    // Print the result
    let result_mapped = result_slice.get_mapped_range();
    let result_data: &[f32] = bytemuck::cast_slice(&*result_mapped);
    //let result_matrix: Matrix4<f32> = Matrix4::from_slice(result_data);
    

    //let result_matrix: Matrix4<f32> = bytemuck::cast_slice(&*result_mapped).into();
    println!("Result Matrix:\n{:?}", result_data);

    // Unmap the buffer
    drop(result_mapped);
    //result_slice.unmap();
	})
}

fn build_ui(app: &Application) {
    let grid = Grid::new();

    let stop_flag = Arc::new(Mutex::new(true));
    let stop_ouput_flag = Arc::new(Mutex::new(true));

    init_device();

    let DropDown1 = DropDown::builder()
        .margin_top(10)
        .margin_bottom(0)
        .margin_start(10)
        .margin_end(10)
        .halign(Align::Start)
        .valign(Align::Center)
        .build();

    let host = cpal::default_host();
    let input_device = host.default_input_device().expect("no default input device");
    let output_device = host.default_output_device().expect("no default input device");

    let label1 = Label::builder().label(&format!("Default input: {}", input_device.name().unwrap())).build();
    let label2 = Label::builder().label(&format!("Default ouput: {}", output_device.name().unwrap())).build();

    let audio_inputs = audio_helper::get_audio_input_devices();

    let list_model = StringList::new(&[]);
    for input in audio_inputs {
        println!("{}", input);
        list_model.append(input)
    }
    DropDown1.set_model(Some(&list_model));
    DropDown1.set_selected(0);

    let button1 = Button::builder()
        .label("Start input")
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .build();

    let button2 = Button::builder()
        .label("Start output")
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .build();


    button1.connect_clicked(move |button| {
        //let selected_id = DropDown1.bo.selected();
        if button.label().unwrap() == "Start input" {
            start_input_stream(&Arc::clone(&stop_flag), 2);
            button.set_label("Stop input");
            *stop_flag.lock().unwrap() = false;
        } else {
            button.set_label("Start input");
            *stop_flag.lock().unwrap() = true;
        }

    });

    button2.connect_clicked(move |button| {
        //let selected_id = DropDown1.bo.selected();
        if button.label().unwrap() == "Start output" {
            start_output_stream(&Arc::clone(&stop_ouput_flag), 0);
            button.set_label("Stop output");
            *stop_ouput_flag.lock().unwrap() = false;
        } else {
            button.set_label("Start output");
            *stop_ouput_flag.lock().unwrap() = true;
        }

    });

    let mut drawing_area = DrawingArea::new();
    drawing_area.set_size_request(512,300);


    drawing_area.set_draw_func(move |_area, cr, width, height| {
        cr.set_source_rgb(0.0, 0.0, 0.0);
        cr.paint();
        cr.set_source_rgb(1.0, 1.0, 1.0);
        cr.set_line_width(2.0);

        let audio_data = GRAPH_DATA.lock().unwrap();

        if let Some(first_sample) = audio_data.first() {
            let height = 150.0 - (*first_sample * 150.0);
            let x_len = 512.0 / audio_data.len() as f64;
            cr.move_to(0.0, height as f64);
            for (i, &sample) in audio_data.iter().enumerate().skip(1) {
                let x = i as f64 * x_len;
                let y = 150.0 - (sample * 150.0);
                cr.line_to(x, y as f64);
            }
        }

        cr.stroke();
    });

    grid.attach(&button1, 0 as i32, 0 as i32, 1, 1);
    grid.attach(&button2, 1 as i32, 0 as i32, 1, 1);
    grid.attach(&label1, 0 as i32, 1 as i32, 1, 1);
    grid.attach(&label2, 1 as i32, 1 as i32, 1, 1);
    grid.attach(&DropDown1, 2 as i32, 0 as i32, 1, 1);
    grid.attach(&drawing_area, 0 as i32, 2 as i32, 1, 1);

    let window = ApplicationWindow::builder()
        .application(app)
        .title("My GTK App")
        .child(&grid)
        .build();
    window.set_default_size(1000, 700);


    glib::timeout_add_local(Duration::from_millis(50), move || {
        drawing_area.queue_draw();
        glib::ControlFlow::Continue
    });

    window.present();
}

fn start_input_stream(stop: &Arc<Mutex<bool>>, selected_id: u32) {
    let stop_flag = Arc::clone(&stop);

    thread::spawn(move || {
        let host = cpal::default_host();
        //let input_device = host.input_devices().unwrap().nth(usize::try_from(selected_id).unwrap()).unwrap();
        let input_device = host
            .default_input_device()
            .expect("no default input device");

        let def_config = input_device.default_input_config().unwrap().config();

        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
        let data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut audio_data = AUDIO_DATA.lock().unwrap();
            let mut graph_data = GRAPH_DATA.lock().unwrap();

            graph_data.clear();
            graph_data.extend_from_slice(&data);
            audio_data.clear();
            audio_data.extend_from_slice(&data);
            return;
        };

        let stream = input_device
            .build_input_stream::<f32, _, _>(&def_config, data_fn, err_fn, None)
            .expect("stream error");
        stream.play().unwrap();

        //thread::sleep(Duration::from_secs(60 * 60 * 24));

        while !*stop_flag.lock().unwrap() {
            thread::sleep(std::time::Duration::from_millis(100));
        }

        drop(stream);

        let mut audio_data = AUDIO_DATA.lock().unwrap();

        audio_data.clear();

        println!("Thread dead...");
    });
}


fn start_output_stream(stop: &Arc<Mutex<bool>>, selected_id: u32) {
    let stop_flag = Arc::clone(&stop);

    thread::spawn(move || {
        let host = cpal::default_host();
        //let input_device = host.input_devices().unwrap().nth(usize::try_from(selected_id).unwrap()).unwrap();
        let input_device = host
            .default_output_device()
            .expect("no default input device");

        let def_config = input_device.default_output_config().unwrap().config();

        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
        let data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut audio_data = AUDIO_DATA.lock().unwrap();

            let len = std::cmp::min(data.len(), audio_data.len());
            //data.copy_from_slice(&audio_data[..len]);
            if len != 0 {
                for i in 0..data.len() {
                    let index = (i as f64 / data.len() as f64 * audio_data.len() as f64).floor() as usize;
                    data[i] = audio_data[index];
                }

                // Remove the processed data from the shared_data buffer
            }

            audio_data.clear();


            //data.copy_from_slice(&*audio_data);

            //audio_data.clear();
            return;
        };

        let stream = input_device
            .build_output_stream::<f32, _, _>(&def_config, data_fn, err_fn, None)
            .expect("stream error");
        stream.play().unwrap();

        //thread::sleep(Duration::from_secs(60 * 60 * 24));

        while !*stop_flag.lock().unwrap() {
            thread::sleep(std::time::Duration::from_millis(100));
        }

        drop(stream);

        let mut audio_data = AUDIO_DATA.lock().unwrap();

        audio_data.clear();

        println!("Thread dead...");
    });
}