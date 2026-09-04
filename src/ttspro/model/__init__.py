"""Inference-side model code: VITS-style synthesizer and speaker encoder.

Must never import ``ttspro.train`` or ``ttspro.data`` (rule 7). Every block is
written to survive ``torch.onnx.export`` on opset 17 using only ops both ORT Web
execution providers run (rule 4, ADR 0005); all randomness comes in as inputs
(rule 5).
"""
