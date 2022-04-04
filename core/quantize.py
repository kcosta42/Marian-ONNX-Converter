import os

from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize(path):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on
    modern CPU

    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures

    Args:
        path: Path to location the exported ONNX model is stored

    Returns:
        The Path generated for the quantized
    """
    print("Quantizing...")

    quantize_dynamic(
        model_input=path,
        model_output=path,
        per_channel=True,
        reduce_range=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
        optimize_model=True,
    )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul'],
    os.remove(path[:-5] + "-opt.onnx")
    print("Done")
