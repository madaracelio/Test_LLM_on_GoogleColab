# Test a LLM on Google Colab

This is the original code that you can test on Google colab : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XwG2dAejKa_dkowpfewv5K1Rd8_qQTq3)

# Details about the code inside the notebook

The following code work on google colab or inside your own local PC

## 1. Install dependencies

On google colab :

```
  !pip install -q transformers SentencePiece accelerate bitsandbytes
```

On local machine (in .py file not .ipynb), juste remove ``!`` during installation on your console.

## 2. Import and prepare the base model name (Llama, Mpt, ...)

You can found the model below here : [OpenLlama_3B](https://huggingface.co/openlm-research/open_llama_3b)

```
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM

  model_path = 'openlm-research/open_llama_3b'
```

## 3. Tokenizer and loading model

```
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False, # True if you have low GPU memory and False if not
    torch_dtype=torch.float16,
    device_map='cuda:0', # Single GPU, and auto for multiple GPU
  )
```

## 4. Test the model

```
  # Query
  prompt = 'Q: What is the largest animal?\nA:'

  # Tokenization
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids

  # Prediction
  generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
  )

  # Response
  print(tokenizer.decode(generation_output[0]))
```
