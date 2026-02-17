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