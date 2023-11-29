// compute_shader.wgsl

[[block]]
struct DataBlock {
    data: array<f32, 4>;
};

[[block]]
struct ResultBlock {
    result: array<f32, 4>;
};

[[group(0), binding(0)]]
var<storage, read_write> data: DataBlock;

[[group(0), binding(1)]]
var<storage, read_write> result: ResultBlock;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    // Read data from buffer
    let inputData: vec4<f32> = vec4<f32>(data.data[0], data.data[1], data.data[2], data.data[3]);

    //let res: array<f32, 4> = [0.0, 0.0, 0.0, 0.0];


    result.result = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
}
