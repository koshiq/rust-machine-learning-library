mod matrix;
mod model;

use matrix::Matrix;
use model::{
    ModelContext, ModelTrainingDesc,
    MV_FLAG_INPUT, MV_FLAG_OUTPUT, MV_FLAG_DESIRED_OUTPUT,
    MV_FLAG_COST, MV_FLAG_REQUIRES_GRAD, MV_FLAG_PARAMETER,
};

fn main() {
    let train_images = Matrix::load(60000, 784, "train_images.mat");
    let mut train_labels = Matrix::new(60000, 10);
    let test_images  = Matrix::load(10000, 784, "test_images.mat");
    let mut test_labels  = Matrix::new(10000, 10);

    let train_labels_raw = Matrix::load(60000, 1, "train_labels.mat");
    let test_labels_raw  = Matrix::load(10000, 1, "test_labels.mat");

    for i in 0..60000 {
        let digit = train_labels_raw.data[i] as usize;
        train_labels.data[i * 10 + digit] = 1.0;
    }
    for i in 0..10000 {
        let digit = test_labels_raw.data[i] as usize;
        test_labels.data[i * 10 + digit] = 1.0;
    }

    draw_mnist_digit(&test_images.data[..784]);
    for i in 0..10 {
        print!("{:.0} ", test_labels.data[i]);
    }
    println!("\n");

    let mut model = ModelContext::new();
    create_mnist_model(&mut model);
    model.compile();

    // pre-training inference
    let input_vi = model.input.unwrap();
    model.vars[input_vi].val.data[..784]
        .copy_from_slice(&test_images.data[..784]);
    model.feedforward();

    print!("pre-training output: ");
    let output_vi = model.output.unwrap();
    for i in 0..10 {
        print!("{:.2} ", model.vars[output_vi].val.data[i]);
    }
    println!();

    model.train(&ModelTrainingDesc {
        train_images:  &train_images,
        train_labels:  &train_labels,
        test_images:   &test_images,
        test_labels:   &test_labels,
        epochs:        10,
        batch_size:    50,
        learning_rate: 0.01,
    });

    // post-training inference
    model.vars[input_vi].val.data[..784]
        .copy_from_slice(&test_images.data[..784]);
    model.feedforward();

    print!("post-training output: ");
    for i in 0..10 {
        print!("{:.4} ", model.vars[output_vi].val.data[i]);
    }
    println!();
}

fn draw_mnist_digit(data: &[f32]) {
    for y in 0..28 {
        for x in 0..28 {
            let val = data[x + y * 28];
            let col = 232 + (val * 23.0) as u32;
            print!("\x1b[48;5;{}m  ", col);
        }
        println!();
    }
    println!("\x1b[0m");
}

fn create_mnist_model(model: &mut ModelContext) {
    let input = model.mv_create(784, 1, MV_FLAG_INPUT);

    let w0 = model.mv_create(16,  784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w1 = model.mv_create(16,  16,  MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w2 = model.mv_create(10,  16,  MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    let bound0 = (6.0f32 / (784.0 + 16.0)).sqrt();
    let bound1 = (6.0f32 / (16.0  + 16.0)).sqrt();
    let bound2 = (6.0f32 / (16.0  + 10.0)).sqrt();
    model.vars[w0].val.fill_rand(-bound0, bound0);
    model.vars[w1].val.fill_rand(-bound1, bound1);
    model.vars[w2].val.fill_rand(-bound2, bound2);

    let b0 = model.mv_create(16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let b1 = model.mv_create(16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let b2 = model.mv_create(10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // layer 0
    let z0_a = model.mv_matmul(w0, input, 0).unwrap();
    let z0_b = model.mv_add(z0_a, b0, 0).unwrap();
    let a0   = model.mv_relu(z0_b, 0);

    // layer 1 (residual)
    let z1_a = model.mv_matmul(w1, a0, 0).unwrap();
    let z1_b = model.mv_add(z1_a, b1, 0).unwrap();
    let z1_c = model.mv_relu(z1_b, 0);
    let a1   = model.mv_add(a0, z1_c, 0).unwrap();

    // layer 2 (output)
    let z2_a  = model.mv_matmul(w2, a1, 0).unwrap();
    let z2_b  = model.mv_add(z2_a, b2, 0).unwrap();
    let output = model.mv_softmax(z2_b, MV_FLAG_OUTPUT);

    let y    = model.mv_create(10, 1, MV_FLAG_DESIRED_OUTPUT);
    let _cost = model.mv_cross_entropy(y, output, MV_FLAG_COST).unwrap();
}
