use rand::Rng;

pub struct Matrix {
    pub rows: u32,
    pub cols: u32,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: u32, cols: u32) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; (rows as usize) * (cols as usize)],
        }
    }

    pub fn load(rows: u32, cols: u32, filename: &str) -> Self {
        let bytes = std::fs::read(filename).expect("failed to read file");
        let max_floats = (rows as usize) * (cols as usize);
        let num_floats = (bytes.len() / 4).min(max_floats);

        let mut data = vec![0.0f32; max_floats];
        for i in 0..num_floats {
            data[i] = f32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        Self { rows, cols, data }
    }

    pub fn copy_from(&mut self, src: &Matrix) -> bool {
        if self.rows != src.rows || self.cols != src.cols {
            return false;
        }
        self.data.copy_from_slice(&src.data);
        true
    }

    pub fn clear(&mut self){
        self.data.fill(0.0);
    }

    pub fn fill(&mut self, x: f32) {
        self.data.fill(x);
    }

    pub fn fill_rand(&mut self, lower: f32, upper: f32) {
        let mut rng = rand::thread_rng();
        for v in &mut self.data {
            *v = rng.gen::<f32>() * (upper - lower) + lower;
        }
    }

    pub fn scale(&mut self, s: f32) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn argmax(&self) -> usize {
        let mut max_i = 0;
        for i in 1..self.data.len() {
            if self.data[i] > self.data[max_i] {
                max_i = i;
            }
        }
        max_i
    }
}

pub fn mat_add(out: &mut Matrix, a: &Matrix, b: &Matrix) -> bool {
    if a.rows != b.rows || a.cols != b.cols { return false; }
    if out.rows != a.rows || out.cols != a.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }
    true
}

pub fn mat_sub(out: &mut Matrix, a: &Matrix, b: &Matrix) -> bool {
    if a.rows != b.rows || a.cols != b.cols { return false; }
    if out.rows != a.rows || out.cols != a.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] - b.data[i];
    }
    true
}

fn mat_mul_nn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    let out_cols = out.cols as usize;
    let a_cols = a.cols as usize;
    let b_cols = b.cols as usize;

    for i in 0..out.rows as usize {
        for k in 0..a_cols {
            for j in 0..out_cols {
                out.data[j + i * out_cols] +=
                    a.data[k + i * a_cols] * b.data[j + k * b_cols];
            }
        }
    }
}

fn mat_mul_nt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    let out_cols = out.cols as usize;
    let a_cols = a.cols as usize;
    let b_cols = b.cols as usize;

    for i in 0..out.rows as usize {
        for j in 0..out_cols {
            for k in 0..a_cols {
                out.data[j + i * out_cols] +=
                    a.data[k + i * a_cols] * b.data[k + j * b_cols];
            }
        }
    }
}

fn mat_mul_tn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    let out_cols = out.cols as usize;
    let a_cols = a.cols as usize;
    let b_cols = b.cols as usize;

    for k in 0..a.rows as usize {
        for i in 0..out.rows as usize {
            for j in 0..out_cols {
                out.data[j + i * out_cols] +=
                    a.data[i + k * a_cols] * b.data[j + k * b_cols];
            }
        }
    }
}

fn mat_mul_tt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    let out_cols = out.cols as usize;
    let a_cols = a.cols as usize;
    let b_cols = b.cols as usize;

    for i in 0..out.rows as usize {
        for j in 0..out_cols {
            for k in 0..a.rows as usize {
                out.data[j + i * out_cols] +=
                    a.data[i + k * a_cols] * b.data[k + j * b_cols];
            }
        }
    }
}

pub fn mat_mul(
    out: &mut Matrix, a: &Matrix, b: &Matrix,
    zero_out: bool, transpose_a: bool, transpose_b: bool,
) -> bool {
    let a_rows = if transpose_a { a.cols } else { a.rows };
    let a_cols = if transpose_a { a.rows } else { a.cols };
    let b_rows = if transpose_b { b.cols } else { b.rows };
    let b_cols = if transpose_b { b.rows } else { b.cols };

    if a_cols != b_rows { return false; }
    if out.rows != a_rows || out.cols != b_cols { return false; }

    if zero_out {
        out.clear();
    }

    match (transpose_a, transpose_b) {
        (false, false) => mat_mul_nn(out, a, b),
        (false, true)  => mat_mul_nt(out, a, b),
        (true,  false) => mat_mul_tn(out, a, b),
        (true,  true)  => mat_mul_tt(out, a, b),
    }

    true
}


pub fn mat_relu(out: &mut Matrix, input: &Matrix) -> bool {
    if out.rows != input.rows || out.cols != input.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] = input.data[i].max(0.0);
    }
    true
}

pub fn mat_softmax(out: &mut Matrix, input: &Matrix) -> bool {
    if out.rows != input.rows || out.cols != input.cols { return false; }

    let mut sum = 0.0f32;
    for i in 0..out.data.len() {
        out.data[i] = input.data[i].exp();
        sum += out.data[i];
    }

    out.scale(1.0 / sum);
    true
}

pub fn mat_cross_entropy(out: &mut Matrix, p: &Matrix, q: &Matrix) -> bool {
    if p.rows != q.rows || p.cols != q.cols { return false; }
    if out.rows != p.rows || out.cols != p.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] = if p.data[i] == 0.0 {
            0.0
        } else {
            p.data[i] * -q.data[i].ln()
        };
    }
    true
}

pub fn mat_relu_add_grad(out: &mut Matrix, input: &Matrix, grad: &Matrix) -> bool {
    if out.rows != input.rows || out.cols != input.cols { return false; }
    if out.rows != grad.rows || out.cols != grad.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] += if input.data[i] > 0.0 { grad.data[i] } else { 0.0 };
    }
    true
}

pub fn mat_softmax_add_grad(out: &mut Matrix, softmax_out: &Matrix, grad: &Matrix) -> bool {
    if softmax_out.rows != 1 && softmax_out.cols != 1 { return false; }

    let size = softmax_out.rows.max(softmax_out.cols) as usize;
    let mut jacobian = Matrix::new(size as u32, size as u32);

    for i in 0..size {
        for j in 0..size {
            let kronecker = if i == j { 1.0 } else { 0.0 };
            jacobian.data[j + i * size] =
                softmax_out.data[i] * (kronecker - softmax_out.data[j]);
        }
    }

    mat_mul(out, &jacobian, grad, false, false, false);
    true
}

pub fn mat_cross_entropy_add_grad(
    p_grad: Option<&mut Matrix>,
    q_grad: Option<&mut Matrix>,
    p: &Matrix, q: &Matrix, grad: &Matrix,
) -> bool {
    if p.rows != q.rows || p.cols != q.cols { return false; }

    let size = p.data.len();

    if let Some(pg) = p_grad {
        if pg.rows != p.rows || pg.cols != p.cols { return false; }
        for i in 0..size {
            pg.data[i] += -q.data[i].ln() * grad.data[i];
        }
    }

    if let Some(qg) = q_grad {
        if qg.rows != q.rows || qg.cols != q.cols { return false; }
        for i in 0..size {
            qg.data[i] += -p.data[i] / q.data[i] * grad.data[i];
        }
    }

    true
}