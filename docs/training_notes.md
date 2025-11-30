# Training Notes

- Input: 12x8x8 planes (piece-centric).
- Label: move index in 0..4095 (from/to squares).
- Model: small ConvNet policy head.
- Loss: NLLLoss over move-space.
