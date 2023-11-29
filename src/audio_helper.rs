use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Data, FromSample, Sample, SampleFormat};

pub fn get_audio_input_devices() -> Vec<&'static str> {
    let mut vec: Vec<&str> = Vec::new();
    let host = cpal::default_host();
    let input_devices = host.input_devices().expect("no input devices");
    for device in input_devices {
        let device_name = device.name().expect("no device name");
        vec.push(Box::leak(device_name.into_boxed_str()));
    }

    return vec;
}