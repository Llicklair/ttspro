"""torch -> ONNX export for the two graphs declared in ``models/contrato.json``.

The exporter reads the contract and refuses to write a graph whose signature
does not match it (rule 2). It then loads the result in onnxruntime and checks
the op list against what ORT Web supports (rule 4).
"""
