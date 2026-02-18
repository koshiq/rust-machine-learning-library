use crate::matrix:: {
    Matrix, mat_add, mat_mul, mat_relu, mat_softmax, mat_cross_entropy,
};
use rand::Rng;

pub const MV_FLAG_REQUIRES_GRAD: u32 = 1 << 0;
pub const MV_FLAG_PARAMETER: u32 = 1 << 1;
pub const MV_FLAG_INPUT: u32 = 1 << 2;
pub const MV_FLAG_OUTPUT: u32 = 1 << 3;
pub const MV_FLAG_DESIRED_OUTPUT: u32 = 1 << 4;
pub const MV_FLAG_COST: u32 = 1 << 5;


#[derive(Clone, Copy, PartialEq, Default)]
pub enum ModelVarOp {
    #[default]
    Null, Create,
    Relu, Softmax,
    Add, MatMul, CrossEntropy,
}

fn num_inputs(op: ModelVarOp) -> usize {
    match op {
        ModelVarOp::Null | ModelVarOp::Create                                               => 0,
        ModelVarOp::Relu | ModelVarOp::Softmax                                              => 1,
        ModelVarOp::Add  | ModelVarOp::MatMul | ModelVarOp::CrossEntropy  => 2,
    }
}

pub struct ModelVar {
    pub flags:  u32,
    pub val:    Matrix,
    pub grad:   Option<Matrix>,
    pub op:     ModelVarOp,
    pub inputs: [Option<usize>; 2],
}

pub struct ModelProgram {
    pub vars: Vec<usize>,
}

impl ModelProgram {
    pub fn empty() -> Self {
        Self { vars: Vec::new() }
    }
}

pub struct ModelContext {
    pub vars:           Vec<ModelVar>,
    pub input:          Option<usize>,
    pub output:         Option<usize>,
    pub desired_output: Option<usize>,
    pub cost:           Option<usize>,
    pub forward_prog:   ModelProgram,
    pub cost_prog:      ModelProgram,
}

pub struct ModelTrainingDesc<'a> {
    pub train_images:   &'a Matrix,
    pub train_labels:   &'a Matrix,
    pub test_images:    &'a Matrix,
    pub test_labels:    &'a Matrix,
    pub epochs:         u32,
    pub batch_size:     u32,
    pub learning_rate:  f32,
}

impl ModelContext {
    pub fn new() -> Self {
        Self {
            vars:           Vec::new(),
            input:          None,
            output:         None,
            desired_output: None,
            cost:           None,
            forward_prog:   ModelProgram::empty(),
            cost_prog:      ModelProgram::empty(),
        }
    }

    pub fn mv_create(&mut self, rows: u32, cols: u32, flags: u32) -> usize {
        let index = self.vars.len();

        self.vars.push(ModelVar {
            flags,
            val: Matrix::new(rows, cols),
            grad: if flags & MV_FLAG_REQUIRES_GRAD != 0 { Some(Matrix::new(rows, cols)) } else { None },
            op: ModelVarOp::Create,
            inputs: [None, None],
        });

        if flags & MV_FLAG_INPUT            != 0 { self.input               = Some(index); }
        if flags & MV_FLAG_OUTPUT           != 0 { self.output              = Some(index); }
        if flags & MV_FLAG_DESIRED_OUTPUT   != 0 { self.desired_output      = Some(index); }
        if flags & MV_FLAG_COST             != 0 { self.cost                = Some(index); }

        index
    }

    fn mv_unary_impl(&mut self, input: usize, rows: u32, cols: u32, mut flags: u32, op: ModelVarOp) -> usize {
        if self.vars[input].flags & MV_FLAG_REQUIRES_GRAD != 0 { flags |= MV_FLAG_REQUIRES_GRAD; }
        let out = self.mv_create(rows, cols, flags);
        self.vars[out].op = op;
        self.vars[out].inputs[0] = Some(input);
        out
    }

    fn mv_binary_impl(&mut self, a: usize, b: usize, rows: u32, cols: u32, mut flags: u32, op: ModelVarOp) -> usize {
        if self.vars[a].flags & MV_FLAG_REQUIRES_GRAD != 0
        || self.vars[b].flags & MV_FLAG_REQUIRES_GRAD != 0 {
            flags |= MV_FLAG_REQUIRES_GRAD;
        }
        let out = self.mv_create(rows, cols, flags);
        self.vars[out].op = op;
        self.vars[out].inputs[0] = Some(a);
        self.vars[out].inputs[1] = Some(b);
        out
    }

    pub fn mv_relu(&mut self, input: usize, flags: u32) -> usize {
        let (r, c) = (self.vars[input].val.rows, self.vars[input].val.cols);
        self.mv_unary_impl(input, r, c, flags, ModelVarOp::Relu)
    }

    pub fn mv_softmax(&mut self, input: usize, flags: u32) -> usize {
        let (r, c) = (self.vars[input].val.rows, self.vars[input].val.cols);
        self.mv_unary_impl(input, r, c, flags, ModelVarOp::Softmax)
    }

    pub fn mv_add(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (ar, ac) = (self.vars[a].val.rows, self.vars[a].val.cols);
        if ar != self.vars[b].val.rows || ac != self.vars[b].val.cols { return None; }
        Some(self.mv_binary_impl(a, b, ar, ac, flags, ModelVarOp::Add))
    }

    pub fn mv_matmul(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (ar, ac) = (self.vars[a].val.rows, self.vars[a].val.cols);
        let (br, bc) = (self.vars[b].val.rows, self.vars[b].val.cols);
        if ac != br { return None; }
        Some(self.mv_binary_impl(a, b, ar, bc, flags, ModelVarOp::MatMul))
    }

    pub fn mv_cross_entropy(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (ar, ac) = (self.vars[a].val.rows, self.vars[a].val.cols);
        if ar != self.vars[b].val.rows || ac != self.vars[b].val.cols { return None; }
        Some(self.mv_binary_impl(a, b, ar, ac, flags, ModelVarOp::CrossEntropy))
    }

    pub fn compile(&mut self) {
        let n = self.vars.len();
        if let Some(i) = self.output { self.forward_prog    = model_prog_create(&self.vars, n, i); }
        if let Some(i) = self.cost   { self.cost_prog       = model_prog_create(&self.vars, n, i); }
    }

    pub fn feedforward(&mut self) {
        let prog: Vec<usize> = self.forward_prog.vars.clone();
        prog_compute(&mut self.vars, &prog);
    }

    pub fn train(&mut self, desc: &ModelTrainingDesc) {
        let num_examples = desc.train_images.rows as usize;
        let input_size   = desc.train_images.cols as usize;
        let output_size  = desc.train_labels.cols as usize;
        let num_tests    = desc.test_images.rows as usize;
        let num_batches  = num_examples / desc.batch_size as usize;

        let mut rng = rand::thread_rng();
        let mut order: Vec<usize> = (0..num_examples).collect();
        let cost_prog: Vec<usize> = self.cost_prog.vars.clone();

        for epoch in 0..desc.epochs as usize {
            for _ in 0..num_examples {
                let a = rng.gen_range(0..num_examples);
                let b = rng.gen_range(0..num_examples);
                order.swap(a, b);
            }

            for batch in 0..num_batches {
                for &vi in &cost_prog {
                    if self.vars[vi].flags & MV_FLAG_PARAMETER != 0 {
                        if let Some(g) = &mut self.vars[vi].grad { g.clear(); }
                    }
                }

                let mut avg_cost = 0.0f32;
                for i in 0..desc.batch_size as usize {
                    let idx        = order[batch * desc.batch_size as usize + i];
                    let input_vi   = self.input.unwrap();
                    let desired_vi = self.desired_output.unwrap();

                    self.vars[input_vi].val.data
                        .copy_from_slice(&desc.train_images.data[idx * input_size .. (idx + 1) * input_size]);
                    self.vars[desired_vi].val.data
                        .copy_from_slice(&desc.train_labels.data[idx * output_size .. (idx + 1) * output_size]);

                    prog_compute(&mut self.vars, &cost_prog);
                    prog_compute_grads(&mut self.vars, &cost_prog);

                    avg_cost += self.vars[self.cost.unwrap()].val.sum();
                }
                avg_cost /= desc.batch_size as f32;

                // gradient descent
                for &vi in &cost_prog {
                    let var = &mut self.vars[vi];
                    if var.flags & MV_FLAG_PARAMETER == 0 { continue; }
                    let scale = desc.learning_rate / desc.batch_size as f32;
                    if let Some(g) = &mut var.grad { g.scale(scale); }
                    let n = var.val.data.len();
                    if let Some(g) = &var.grad {
                        let gdata: Vec<f32> = g.data.clone();
                        for i in 0..n { var.val.data[i] -= gdata[i]; }
                    }
                }

                use std::io::Write;
                print!("\rEpoch {:2}/{:2}  Batch {:4}/{:4}  Cost: {:.4}",
                    epoch + 1, desc.epochs, batch + 1, num_batches, avg_cost);
                std::io::stdout().flush().unwrap();
            }
            println!();

            // evaluate on test set
            let output_vi  = self.output.unwrap();
            let desired_vi = self.desired_output.unwrap();
            let cost_vi    = self.cost.unwrap();
            let (mut correct, mut avg_cost) = (0usize, 0.0f32);

            for i in 0..num_tests {
                let input_vi = self.input.unwrap();
                self.vars[input_vi].val.data
                    .copy_from_slice(&desc.test_images.data[i * input_size .. (i + 1) * input_size]);
                self.vars[desired_vi].val.data
                    .copy_from_slice(&desc.test_labels.data[i * output_size .. (i + 1) * output_size]);

                prog_compute(&mut self.vars, &cost_prog);

                avg_cost += self.vars[cost_vi].val.sum();
                if self.vars[output_vi].val.argmax() == self.vars[desired_vi].val.argmax() {
                    correct += 1;
                }
            }

            println!("Accuracy: {}/{} ({:.1}%)  Cost: {:.4}",
                correct, num_tests,
                correct as f32 / num_tests as f32 * 100.0,
                avg_cost / num_tests as f32);
        }
    }
}

fn model_prog_create(vars: &[ModelVar], num_vars: usize, root: usize) -> ModelProgram {
    let mut visited = vec![false; num_vars];
    let mut stack   = Vec::<usize>::with_capacity(num_vars);
    let mut out     = Vec::<usize>::with_capacity(num_vars);

    stack.push(root);

    while let Some(&cur_idx) = stack.last() {
        if visited[cur_idx] {
            stack.pop();
            out.push(cur_idx);
            continue;
        }

        visited[cur_idx] = true;

        let n = num_inputs(vars[cur_idx].op);
        for i in 0..n {
            if let Some(inp) = vars[cur_idx].inputs[i] {
                if inp < num_vars && !visited[inp] {
                    stack.retain(|&x| x != inp);
                    stack.push(inp);
                }
            }
        }
    }

    ModelProgram { vars: out }
}

//Forward Pass
fn prog_compute(vars: &mut [ModelVar], prog: &[usize]) {
    for &cur_idx in prog {
        let op    = vars[cur_idx].op;
        let a_idx = vars[cur_idx].inputs[0];
        let b_idx = vars[cur_idx].inputs[1];

        unsafe {
            let ptr = vars.as_mut_ptr();
            let cur = &mut *ptr.add(cur_idx);

            match op {
                ModelVarOp::Null | ModelVarOp::Create => {}

                ModelVarOp::Relu => {
                    let a = &*ptr.add(a_idx.unwrap());
                    mat_relu(&mut cur.val, &a.val);
                }
                ModelVarOp::Softmax => {
                    let a = &*ptr.add(a_idx.unwrap());
                    mat_softmax(&mut cur.val, &a.val);
                }
                ModelVarOp::Add => {
                    let a = &*ptr.add(a_idx.unwrap());
                    let b = &*ptr.add(b_idx.unwrap());
                    mat_add(&mut cur.val, &a.val, &b.val);
                }
                ModelVarOp::MatMul => {
                    let a = &*ptr.add(a_idx.unwrap());
                    let b = &*ptr.add(b_idx.unwrap());
                    mat_mul(&mut cur.val, &a.val, &b.val, true, false, false);
                }
                ModelVarOp::CrossEntropy => {
                    let a = &*ptr.add(a_idx.unwrap());
                    let b = &*ptr.add(b_idx.unwrap());
                    mat_cross_entropy(&mut cur.val, &a.val, &b.val);
                }
            }
        }
    }
}

//Backward Pass
fn prog_compute_grads(vars: &mut [ModelVar], prog: &[usize]) {
    for &vi in prog {
        if vars[vi].flags & MV_FLAG_REQUIRES_GRAD == 0 { continue; }
        if vars[vi].flags & MV_FLAG_PARAMETER      != 0 { continue; }
        if let Some(g) = &mut vars[vi].grad { g.clear(); }
    }

    if let Some(&last) = prog.last() {
        if let Some(g) = &mut vars[last].grad { g.fill(1.0); }
    }

    for &cur_idx in prog.iter().rev() {
        let op    = vars[cur_idx].op;
        let flags = vars[cur_idx].flags;
        let a_idx = vars[cur_idx].inputs[0];
        let b_idx = vars[cur_idx].inputs[1];

        if flags & MV_FLAG_REQUIRES_GRAD == 0 { continue; }

        let n = num_inputs(op);
        if n >= 1 && vars[a_idx.unwrap()].flags & MV_FLAG_REQUIRES_GRAD == 0
                  && (n < 2 || vars[b_idx.unwrap()].flags & MV_FLAG_REQUIRES_GRAD == 0) {
            continue;
        }

        let cur_grad: Vec<f32> = vars[cur_idx].grad.as_ref().unwrap().data.clone();
        let cur_val:  Vec<f32> = vars[cur_idx].val.data.clone();

        unsafe {
            let ptr = vars.as_mut_ptr();

            match op {
                ModelVarOp::Null | ModelVarOp::Create => {}

                ModelVarOp::Relu => {
                    let a = &mut *ptr.add(a_idx.unwrap());
                    if a.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        let g = a.grad.as_mut().unwrap();
                        for i in 0..g.data.len() {
                            g.data[i] += if a.val.data[i] > 0.0 { cur_grad[i] } else { 0.0 };
                        }
                    }
                }
                ModelVarOp::Softmax => {
                    let a = &mut *ptr.add(a_idx.unwrap());
                    if a.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        let size = cur_val.len();
                        let mut jac = Matrix::new(size as u32, size as u32);
                        for i in 0..size {
                            for j in 0..size {
                                let kron = if i == j { 1.0 } else { 0.0 };
                                jac.data[j + i * size] = cur_val[i] * (kron - cur_val[j]);
                            }
                        }
                        let grad_mat = Matrix { rows: size as u32, cols: 1, data: cur_grad.clone() };
                        mat_mul(a.grad.as_mut().unwrap(), &jac, &grad_mat, false, false, false);
                    }
                }
                ModelVarOp::Add => {
                    let a = &mut *ptr.add(a_idx.unwrap());
                    let b = &mut *ptr.add(b_idx.unwrap());
                    if a.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        let g = a.grad.as_mut().unwrap();
                        for i in 0..g.data.len() { g.data[i] += cur_grad[i]; }
                    }
                    if b.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        let g = b.grad.as_mut().unwrap();
                        for i in 0..g.data.len() { g.data[i] += cur_grad[i]; }
                    }
                }
                ModelVarOp::MatMul => {
                    let a_val: Vec<f32> = (*ptr.add(a_idx.unwrap())).val.data.clone();
                    let b_val: Vec<f32> = (*ptr.add(b_idx.unwrap())).val.data.clone();
                    let a_rows = (*ptr.add(a_idx.unwrap())).val.rows;
                    let a_cols = (*ptr.add(a_idx.unwrap())).val.cols;
                    let b_cols = (*ptr.add(b_idx.unwrap())).val.cols;

                    let cg = Matrix { rows: a_rows, cols: b_cols, data: cur_grad.clone() };
                    let bv = Matrix { rows: a_cols, cols: b_cols, data: b_val };
                    let av = Matrix { rows: a_rows, cols: a_cols, data: a_val };

                    let a = &mut *ptr.add(a_idx.unwrap());
                    if a.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        // a.grad += cur_grad @ b.val^T
                        mat_mul(a.grad.as_mut().unwrap(), &cg, &bv, false, false, true);
                    }
                    let b = &mut *ptr.add(b_idx.unwrap());
                    if b.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        // b.grad += a.val^T @ cur_grad
                        mat_mul(b.grad.as_mut().unwrap(), &av, &cg, false, true, false);
                    }
                }
                ModelVarOp::CrossEntropy => {
                    let p_val: Vec<f32> = (*ptr.add(a_idx.unwrap())).val.data.clone();
                    let q_val: Vec<f32> = (*ptr.add(b_idx.unwrap())).val.data.clone();
                    let size = p_val.len();

                    let a = &mut *ptr.add(a_idx.unwrap());
                    let b = &mut *ptr.add(b_idx.unwrap());

                    if a.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        let g = a.grad.as_mut().unwrap();
                        for i in 0..size { g.data[i] += -q_val[i].ln() * cur_grad[i]; }
                    }
                    if b.flags & MV_FLAG_REQUIRES_GRAD != 0 {
                        let g = b.grad.as_mut().unwrap();
                        for i in 0..size { g.data[i] += -p_val[i] / q_val[i] * cur_grad[i]; }
                    }
                }
            }
        }
    }
}
