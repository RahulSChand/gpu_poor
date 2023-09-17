# Can your GPU run this?

Calculate how much GPU memory you need &amp; breakdown of where it goes for training/inference of any LLM model with quantization (GGML/bnb) & inference frameworks (vLLM/llama.cpp/HF): http://rahulschand.github.io/gpu_poor/


<img width="1157" alt="Screenshot 2023-09-16 at 2 59 34 AM" src="https://github.com/RahulSChand/gpu_poor/assets/16897807/30105eb7-50cf-4bc2-8f73-8e7aedbb48bd">



### Purpose
I made this after a few days of frustation of not being able to finetune a 7b-hf with bnb int8 quanization & sequence length=1000 on a 24GB 4090. This might be useful to people starting out or trying to figure out which LLMs they can train/run on their own GPUs. There are infernece frameworks like GGML which allow you to run LLMs on your CPU (or CPU+GPU) so this is not useful to people that are trying to find the cheapest way to run a particular LLM (which is CPU with ggml).

### How to use

#### Model Name/ID/Size

1. You can either upload the model id of a huggingface model (e.g. meta-llama/Llama-2-7b). Currently I have hardcoded & saved configs of top 3k most downlaoded LLMs on huggingface. 
2. If you have a custom model or your hugginface id isn't available then you can either upload a json config ([example]( https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json)) or just enter your model size (e.g. 7 billion for llama-2-7b)

#### Options
1. **Inference**: Find vRAM for inference using either HuggingFace implementation or vLLM or GGML
2. **Training** : Find vRAM for either full model finetuning or finetuning using LoRA (currently I have hardcoded r=8 for LoRA config) 

#### Quantization
1. Currently it supports: bitsandbytes (bnb) int8/int4 & GGML (QK_8, QK_5, QK_4). The latter are only for inference while bnb int8/int4 can be used for both training & inference

#### Context Len/Sequence Length
1. What is the length of your prompt+new maximum tokens generated. Or for training this is the sequence length of your training data. Batch sizes are 1. The option to specify batch sizes needs to be added.

#### Output
The output is the total vRAM & the breakdown of where the vRAM goes (in MB). It looks like below

```     
{
  Total: 4000,
  "KV Cache": 1000,
  "Model Size": 2500,
  "Activation Memory": 0,
  "Grad & Optimizer memory": 0,
  "cuda + other overhead":  500
}
```

#### Why are the results wrong?
The results can vary depending on your model, input data, cuda version & what quant you are using & it is impossible to predict exact values. I have tried to take these into account & make sure the results arr withing 500MB. Sometimes the answers might be very wrong in which case please open an issue here: https://github.com/RahulSChand/gpu_poor
