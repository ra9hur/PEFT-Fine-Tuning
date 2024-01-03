## Optimizing LLMs: A Step-by-Step Guide to Fine-Tuning with PEFT - QLoRA

Below tables [1](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/LLM3.png), [2](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/) offers a summary of large language models, including original release date, largest model size, and whether the weights are fully open source to the public:

![LLMs_by_size](https://github.com/ra9hur/PEFT-Fine-Tuning/assets/17127066/94c0ded6-6a69-4658-80b5-6ce677bd6b84)

Language models are becoming larger all the time. At the time of this writing, PaLM has 540B parameters, OPT, GPT-3, and BLOOM have around 176B parameters, and we are trending towards even larger models.

These models are hard to run on easily accessible devices. For example, just to do inference on BLOOM-176B, you would need to have 8x 80GB A100 GPUs (~$15k each). To fine-tune BLOOM-176B, you'd need 72 of these GPUs! Much larger models, like PaLM would require even more resources.

Because these huge models require so many GPUs to run, we need to find ways to reduce these requirements while preserving the model's performance. Various technologies have been developed that try to shrink the model size - quantization and distillation, and there are many others.

In this implementation, we have tried to load an LLM, apply PEFT-QLoRA and then further fine-tune using an instruction-following dataset.

## Packages to be installed

- Bits And Bytes
- Transformers
- PEFT
- Accelerate

#### Bits And Bytes

QLoRA uses 4-bit quantization to compress a pre-trained language model. Bits and bytes quantization Library is used to achieve this.

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                quantization_config=bnb_config, 
                                                device_map={"":0})


4-bit NormalFloat4 is an optimized data type that can be used to store weights, which brings down the memory footprint considerably. 4-bit NormalFloat4 quantization is a 3-step process.

Double quantization, further reduces the memory footprint, by quantizing, quantization constants.


#### PEFT (parameter efficient fine tuning)

![PEFT](https://github.com/ra9hur/PEFT-Fine-Tuning/assets/17127066/0c37623b-f4c4-4c77-92b6-4cd1fe2eaa81)

It involves a new wave of machine learning that allow us to tweak pre-trained language models for different applications without adjusting all the model parameters. This approach is handy because fine-tuning large LLMs can be extremely resource intensive.

Therefore by fine-tuning only a fraction of the model's parameters PEFT methods significantly reduce the computational and storage costs.

Some of the popular PEFT methods include 
- LoRA
- Prefix Tuning
- P-Tuning
- QLoRA

LoRA is an innovative technique designed to efficiently fine-tune pre-trained language models by injecting trainable low-rank matrices into each layer of the Transformer architecture. LoRA aims to reduce the number of trainable parameters and the computational burden while maintaining or improving the model’s performance on downstream tasks.

QLoRA is an extension of LoRA that further introduces quantization to enhance parameter efficiency during fine-tuning. It builds on the principles of LoRA while introducing 4-bit NormalFloat (NF4) quantization and Double Quantization techniques.


#### Accelerate

Accelerate is a user-friendly tool designed to provide developers with the flexibility of writing their own training Loops for pytorch models while avoiding the hassle of dealing with the extra code required
for Multi-Device setups 

Think of accelerate as your personal assistant taking care of all the tedious parts of your code associated with running your model on different kinds of devices whether there are multiple gpus TPUs or even enabling mixed Precision in practice 

This means that you can add just a handful of lines to your typical pytorch training script and then your model can run smoothly on a variety of setups ranging from a single CPU or GPU to multiple CPUs or  gpus and even TPUs

Even better you can switch between different environments without modifying your code which is super handy for debugging on your local machine before moving to larger scale training setup


## Model and Instruction-Following Dataset

Model: [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m)

Instruction-Response Dataset: [kotzeje/lamini_docs.jsonl](https://huggingface.co/datasets/kotzeje/lamini_docs.jsonl)

## References:

1. [Paper](https://arxiv.org/pdf/2305.14314.pdf)

2. [Colab notebook reference](https://colab.research.google.com/drive/1Vvju5kOyBsDr7RX_YAvp6ZsSOoSMjhKD?usp=sharing)

3. [How to Fine-Tune Open-Source LLMs Locally Using QLoRA!](https://www.youtube.com/watch?v=2bkrL2ZcOiM)

4. [Paper explanation: AI Breakthrough Democratizes Open-Source LLMs! Introducing QLoRA and the Guanaco Family](https://www.youtube.com/watch?v=n90tKMDQUaY)