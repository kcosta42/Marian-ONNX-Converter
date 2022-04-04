# Marian-ONNX

## Instructions

1. Make sure you have the python dependencies installed.

  ```sh
  python3 -m pip install -r requirements.txt
  ```

2. Download a Marian model from huggingface hub (you may need to install `git-lfs`)

  ```sh
  git lfs clone https://huggingface.co/Helsinki-NLP/opus-mt-fr-en ./models/fr-en
  ```

3. Convert the Marian model to ONNX.

  ```sh
  python3 convert.py ./models/fr-en
  ```

## Usage

```py
from transformers import MarianTokenizer

from core.marian import MarianOnnx

DEVICE = 'cpu'
SENTENCES = ["Bonjour", "Je m'appelle Bob"]
MODEL_PATH = "./outs/fr-en"

tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)

input_ids = tokenizer(SENTENCES, return_tensors='pt', padding=True).to(DEVICE)
model = MarianOnnx(MODEL_PATH, device=DEVICE)
tokens = model.generate(**input_ids)
print(tokenizer.batch_decode(tokens, skip_special_tokens=True))
# ['Hello.', 'My name is Bob.']
```

## Benchmark

```sh
# Benchmark for the opus-mt-fr-en model on a NVIDIA GeForce RTX 2070
$> python3 -m core.benchmark

CPU Benchmark:

Warming up ORT...
ORT CPU: 48 ms / sentence
PyTorch CPU: 152 ms / sentence

-----
GPU Benchmark:

Warming up ORT...
ORT GPU: 31 ms / sentence
PyTorch GPU: 66 ms / sentence
```
