import os
import warnings
warnings.filterwarnings('ignore')

import shutil
import argparse

from core.utils import generate_onnx_graph
from core.benchmark import verify_export


PARAMS = None
FILES = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input model directory or name.")
    parser.add_argument("-o", "--output", type=str, default="./outs", help="Output directory.")
    parser.add_argument("--no-quantize", action="store_false", default=True,
                        help="Disable model quantization.")
    return parser.parse_args()


def main():
    outdir = os.path.join(PARAMS.output, os.path.basename(PARAMS.input))
    os.makedirs(outdir, exist_ok=True)

    encoder_path = os.path.join(outdir, "encoder.onnx")
    decoder_path = os.path.join(outdir, "decoder.onnx")

    generate_onnx_graph(PARAMS.input, encoder_path, decoder_path, outdir, quant=PARAMS.no_quantize)

    try:
        verify_export(PARAMS.input, outdir)
    except Exception as e:
        print(e)

    print("Creating archive file...")
    shutil.make_archive(outdir, format="zip", root_dir=outdir)
    print("Done.")


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
