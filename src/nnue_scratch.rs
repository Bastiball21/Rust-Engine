
use crate::nnue::{L1_SIZE, L2_SIZE, L3_SIZE, L4_SIZE};

#[repr(align(64))]
pub struct NNUEScratch {
    pub hidden_l1: [i16; L1_SIZE * 2],
    pub l2_out: [i16; L2_SIZE],
    pub l3_out: [i16; L3_SIZE],
    pub l4_out: [i16; L4_SIZE],
    pub final_out: [i16; 1],
}

impl Default for NNUEScratch {
    fn default() -> Self {
        Self {
            hidden_l1: [0; L1_SIZE * 2],
            l2_out: [0; L2_SIZE],
            l3_out: [0; L3_SIZE],
            l4_out: [0; L4_SIZE],
            final_out: [0; 1],
        }
    }
}
