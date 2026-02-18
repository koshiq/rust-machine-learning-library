use crate::matrix:: {
    Matrix, mat_add, mat_sub, mat_mul, mat_relu, mat_softmax, mat_cross_entropy,
    mat_relu_add_grad, mat_softmax_add_grad, mat_cross_entropy_add_grad,
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
    Add, Sub, MatMul, CrossEntropy,
}

fn num_inputs(op: ModelVarOp) -> usize {
    match op {
        ModelVarOp::Null | ModelVarOp::Create                                               => 0,
        ModelVarOp::Relu | ModelVarOp::Softmax                                              => 1,
        ModelVarOp::Add  | ModelVarOp::Sub | ModelVarOp::MatMul | ModelVarOp::CrossEntropy  => 2,
    }
}

pub struct ModelContext {
    pub vars:           Vec<ModelVar>,
    pub input:          Option<usize>,
    pub ouput:          Option<usize>,
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
    pub batch_sizes:    u32,
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
            index,
            flags,
            val: Matrix::new(rows, cols),
            grad: if flags & MV_FLAG_REQUIRES_GRAD != 0 { Some(Matrix::new(rows, cols)) } else { None },
            op: ModelVarOp::create,
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
        self.mv_unary_impl(input, r, c, flags, ModelVarOp::Relu)
    }

    pub fn mv_add(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (ar, ac) = (self.vars[a].val.rows, self.vars[a].val.cols);
        if ar != self.vars[b].val.rows || ac != self.vars[b].val.cols { return None; }
        Some(self.mv_binary_impl(a, b, ar, ac, flags, ModelVarOp::Add))
    }

    pub fn mv_sub(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (ar, ac) = (self.var[a].val.rows, self.var[a].val.cols);
        if ar != self.vars[b].val.rows || ac != self.vars[b].val.cols { return None; }
        Some(self.mv_binary_impl(a, b, ar, ac, flags, ModelVarOp::Sub))
    }

    pub fn mv_matmul(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (ar, ac) = (self.var[a].val.rows, self.var[a].val.cols);
        let (br, bc) = (self.var[b].val.rows, self.var[b].val.cols);
        if ac != br { return None; }
        Some(self.mv_binary_impl(p, q, pr, pc, flags, ModelVarOp::MatMul))
    }

    pub fn mv_cross_entropy(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (pr, pc) = (self.vars[p].val.rows, self.vars[p].val.cols);
        if pr != self.vars[q].val.rows || pc != self.vars[q].val.cols { return None; }
        Some(self.mv_binary_impl(p, q, pr, pc, flags, ModelVarOp::CrossEntropy))
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

    pub fn train(&mut self, desc: &ModelTrainDesc) {
        let num_examples = desc.train_images.rows as usize;
    }
}