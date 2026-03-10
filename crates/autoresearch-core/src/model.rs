#![cfg(feature = "train")]

use candle_core::{Result, Tensor, DType, Device, Var};
use candle_nn::{
    embedding, layer_norm, linear, Activation, Embedding, LayerNorm, Linear, Module, VarBuilder,
    VarMap,
};

use crate::config::{CoreConstants, TrainConfig};

#[derive(Debug, Clone)]
pub struct GptModelConfig {
    pub seq_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
}

impl GptModelConfig {
    pub fn from_train_config(
        train: &TrainConfig,
        constants: CoreConstants,
        vocab_size: usize,
    ) -> Result<Self> {
        let base_dim = train.depth * train.aspect_ratio;
        let n_embd = base_dim.div_ceil(train.head_dim) * train.head_dim;
        let n_head = n_embd / train.head_dim;

        Ok(Self {
            seq_len: constants.max_seq_len,
            vocab_size,
            n_layer: train.depth,
            n_head,
            n_embd,
        })
    }
}

struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn new(cfg: &GptModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let q_proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("q"))?;
        let k_proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("k"))?;
        let v_proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("v"))?;
        let out_proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_head: cfg.n_head,
            head_dim: cfg.n_embd / cfg.n_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?;

        let kt = k.transpose(2, 3)?;
        let mut att = q.matmul(&kt)?;
        att = att.affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;

        let mask = causal_mask(t, x.device())?;
        att = att.broadcast_add(&mask)?;

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, c))?;
        self.out_proj.forward(&y)
    }
}

struct Mlp {
    fc: Linear,
    proj: Linear,
}

impl Mlp {
    fn new(cfg: &GptModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let fc = linear(cfg.n_embd, cfg.n_embd * 4, vb.pp("fc"))?;
        let proj = linear(cfg.n_embd * 4, cfg.n_embd, vb.pp("proj"))?;
        Ok(Self { fc, proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc.forward(x)?;
        let x = Activation::Gelu.forward(&x)?;
        self.proj.forward(&x)
    }
}

struct Block {
    ln1: LayerNorm,
    ln2: LayerNorm,
    attn: CausalSelfAttention,
    mlp: Mlp,
}

impl Block {
    fn new(cfg: &GptModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let ln1 = layer_norm(cfg.n_embd, 1e-5, vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.n_embd, 1e-5, vb.pp("ln2"))?;
        let attn = CausalSelfAttention::new(cfg, vb.pp("attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            ln1,
            ln2,
            attn,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.attn.forward(&self.ln1.forward(x)?)?;
        let x = (x + a)?;
        let m = self.mlp.forward(&self.ln2.forward(&x)?)?;
        x + m
    }
}

pub struct GptModel {
    config: GptModelConfig,
    var_map: VarMap,
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GptModel {
    pub fn new(config: GptModelConfig, device: &Device) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let token_emb = embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let pos_emb = embedding(config.seq_len, config.n_embd, vb.pp("wpe"))?;
        let blocks = (0..config.n_layer)
            .map(|idx| Block::new(&config, vb.pp(format!("h.{idx}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(config.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            config,
            var_map,
            token_emb,
            pos_emb,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward_logits(&self, idx: &Tensor) -> Result<Tensor> {
        let (b, t) = idx.dims2()?;
        if t > self.config.seq_len {
            candle_core::bail!(
                "input sequence length {} exceeds model max {}",
                t,
                self.config.seq_len
            );
        }

        let tok = self.token_emb.forward(idx)?;
        let pos_ids = Tensor::arange(0u32, t as u32, idx.device())?;
        let pos = self.pos_emb.forward(&pos_ids)?.unsqueeze(0)?;
        let mut x = tok.broadcast_add(&pos)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;

        // Keep batch dimension explicit for callers.
        logits.reshape((b, t, self.config.vocab_size))
    }

    pub fn forward_loss(&self, idx: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(idx)?;
        let (b, t, v) = logits.dims3()?;
        let flat_logits = logits.reshape((b * t, v))?;
        let flat_targets = targets.reshape((b * t,))?;
        candle_nn::loss::cross_entropy(&flat_logits, &flat_targets)
    }

    pub fn variables(&self) -> Vec<Var> {
        self.var_map.all_vars()
    }

    pub fn num_parameters(&self) -> usize {
        self.variables().iter().map(|var| var.elem_count()).sum()
    }

    pub fn config(&self) -> &GptModelConfig {
        &self.config
    }
}

fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut values = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            values[i * seq_len + j] = -1e9;
        }
    }
    Tensor::from_vec(values, (1, 1, seq_len, seq_len), device)
}
