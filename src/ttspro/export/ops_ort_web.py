"""Operators ONNX Runtime Web executes on its WebGPU provider (rule 4, ADR 0005).

Copied from
https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md
on 2026-09-04. Anything a graph uses outside this set falls back to CPU op by
op with a memory copy each way — it does not fail, it gets slow in silence,
which is exactly what the export test exists to catch. Refresh this list when
`onnxruntime-web` is bumped in web/package.json, and record the diff in
docs/evidencia.md.

Notably ABSENT: STFT, RandomNormal, Loop, Scan, GRU, LSTM, TopK, NonZero,
Softplus. Present: Conv, ConvTranspose, Pad, Resize, LayerNormalization,
InstanceNormalization, Einsum, Gather, ScatterND, CumSum, Range, Where, If.
"""

WEBGPU_2026_09 = frozenset(
    """
    Abs Acos Acosh Add ArgMax ArgMin Asin Asinh Atan Atanh Attention AveragePool
    BatchNormalization BiasAdd BiasSplitGelu Cast Ceil Clip Concat Conv ConvTranspose Cos Cosh
    CumSum DFT DepthToSpace DequantizeLinear Div Einsum Elu Equal Erf Exp Expand FastGelu Flatten
    Floor FusedConv Gather GatherBlockQuantized GatherElements GatherND Gelu Gemm GlobalAveragePool
    GlobalMaxPool Greater GreaterOrEqual GridSample GroupQueryAttention HardSigmoid HardSwish If
    InstanceNormalization LayerNormalization LeakyRelu Less LessOrEqual Log MatMul MatMulNBits
    MaxPool MemcpyFromHost MemcpyToHost Mul MultiHeadAttention Neg Not Pad Pow QuickGelu Range
    Reciprocal ReduceL1 ReduceL2 ReduceLogSum ReduceLogSumExp ReduceMax ReduceMean ReduceMin
    ReduceProd ReduceSum ReduceSumSquare Relu Reshape Resize RotaryEmbedding ScatterND Shape
    Sigmoid SimplifiedLayerNormalization Sin Sinh SkipLayerNormalization
    SkipSimplifiedLayerNormalization Slice Softmax Split Sqrt Squeeze Sub Tan Tanh ThresholdedRelu
    Tile Transpose Unsqueeze Where
    """.split()
)

# Shape-only ops ORT resolves at load time; they never reach a kernel and are
# not a fallback even though the list above does not name all of them.
SIN_KERNEL = frozenset({"Constant", "ConstantOfShape", "Identity"})


def ops_fuera_de_webgpu(modelo) -> dict[str, int]:
    """{op_type: count} of nodes in an onnx.ModelProto that WebGPU would not run."""
    fuera: dict[str, int] = {}
    for nodo in modelo.graph.node:
        if nodo.op_type not in WEBGPU_2026_09 and nodo.op_type not in SIN_KERNEL:
            fuera[nodo.op_type] = fuera.get(nodo.op_type, 0) + 1
    return fuera
