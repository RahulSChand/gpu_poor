# Can my GPU run this LLM?

![Made with](https://img.shields.io/badge/logo-javascript-blue?logo=javascript)

Calculate how much GPU memory you need &amp; breakdown of where it goes for training/inference of any LLM model with quantization (GGML/bnb) & inference frameworks (vLLM/llama.cpp/HF). Link: **http://rahulschand.github.io/gpu_poor/**



![smaller_gif-2](https://github.com/RahulSChand/gpu_poor/assets/16897807/980047e9-cf89-4764-8576-aaf842ea83d1)





### Purpose

I made this to check if you can run a particular LLM on your GPU. Useful to figure out the following
1. What quantization I should use
2. What max context length my GPU can handle
3. What max batch size I can use during finetuning
4. What is consuming my GPU memory? What should I change to fit the LLM on my GPU

The output is the total vRAM & the breakdown of where the vRAM goes (in MB). It looks like below

```     
{
  "Total": 4000,
  "KV Cache": 1000,
  "Model Size": 2000,
  "Activation Memory": 500,
  "Grad & Optimizer memory": 0,
  "cuda + other overhead":  500
}
```
### Can't we just look at the model size & figure this out?

Finding which LLMs your GPU can handle isn't as easy as looking at the model size because during inference (KV cache) takes susbtantial amount of memory. For example, with sequence length 1000 on llama-2-7b it takes 1GB of extra memory (using hugginface LlamaForCausalLM, with exLlama & vLLM this is 500MB). And during training both KV cache & activations & quantization overhead take a lot of memory. For example, llama-7b with bnb int8 quant is of size ~7.5GB but it isn't possible to finetune it using LoRA on data with 1000 context length even with RTX 4090 24 GB. Which means an additional 16GB memory goes into quant overheads, activations & grad memory.
 
### How to use

#### Model Name/ID/Size

1. You can either enter the model id of a huggingface model (e.g. meta-llama/Llama-2-7b). Currently I have hardcoded & saved model configs of top 3k most downlaoded LLMs on huggingface.
2. If you have a custom model or your hugginface id isn't available then you can either upload a json config ([example]( https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json)) or just enter your model size (e.g. 7 billion for llama-2-7b)

#### Options
1. **Inference**: Find vRAM for inference using either HuggingFace implementation or vLLM or GGML
2. **Training** : Find vRAM for either full model finetuning or finetuning using LoRA (currently I have hardcoded r=8 for LoRA config) 

#### Quantization
1. Currently it supports: bitsandbytes (bnb) int8/int4 & GGML (QK_8, QK_6, QK_5, QK_4, QK_2). The latter are only for inference while bnb int8/int4 can be used for both training & inference

#### Context Len/Sequence Length
1. What is the length of your prompt+new maximum tokens generated. Or for training this is the sequence length of your training data. Batch sizes are 1 for inference & can be specified for training. The option to specify batch sizes for inference needs to be added.




### How reliable are the numbers?
The results can vary depending on your model, input data, cuda version & what quant you are using & it is impossible to predict exact values. I have tried to take these into account & make sure the results arr withing 500MB. I have cross checked 3b,7b & 13b models against what the website gives & what I get on my RTX 4090 & 2060. Below is the table, all numbers are within 500MB.

<img width="604" alt="image" src="https://github.com/RahulSChand/gpu_poor/assets/16897807/3d49a422-f174-4537-b5fa-42adc4b15a89">


### Why are the results wrong?
Sometimes the answers might be very wrong in which case please open an issue here & I will try to fix it.

### TODO
1. Add support for exLlama
2. Add QLora
3. Add way to measure approximste tokens/s you can get for a particular GPU
