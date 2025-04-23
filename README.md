# 📊 Medical Q&A Fine-tuning Pipeline

In this workshop, you'll learn the end-to-end workflow for creating a synthetic medical Q&A dataset, setting up and fine-tuning a Large Language Model (LLM) using Parameter-Efficient Fine-Tuning (PEFT) techniques, and deploying models efficiently for optimized inference.

We'll guide you through:

- Preparing model checkpoints
- Generating synthetic medical Q&A datasets
- Fine-tuning with NVIDIA NeMo Framework
- Deploying optimized models for inference using NVIDIA Triton and TensorRT-LLM library

> ✨ **Note:** All applications and services are designed to run on the **[UCloud - Interactive HPC](https://docs.cloud.sdu.dk/)** platform.

---

## 📚 Project Notebooks

### 📘 [`01-nemo-medchat-qa-peft.ipynb`](01-nemo-medchat-qa-peft.ipynb)

**Goal:**  
Demonstrate how to perform **PEFT** on an open-source LLM—**Llama 3.1 8B Instruct**—using **NVIDIA NeMo Framework**.

Highlights:
- Convert the base model to **NeMo format** for training compatibility
- Download and preprocess the **MedChat-QA** dataset (from Hugging Face)
- Set up **LoRA** fine-tuning
- Validate the environment and configuration for fine-tuning

---

### 📘 [`02-triton-trtllm-model-deployment.ipynb`](02-triton-trtllm-model-deployment.ipynb)

**Goal:**  
Show how to deploy a LLM using **Triton Inference Server** and **TensorRT-LLM** for optimized inference.

Highlights:
- Convert the **Llama 3.3 70B Instruct** model checkpoint to **TensorRT-LLM format**
- Configure **Triton Inference Server** with **inflight batching** for high-throughput inference
- Measure performance metrics like **latency** and **throughput** using **GenAI-Perf**

---

### 📘 [`03-ls-create-medal-qa-dataset.ipynb`](03-ls-create-medal-qa-dataset.ipynb)

**Goal:**  
Generate a **synthetic medical Q&A dataset** using **Label Studio** connected to a live inference server.

Highlights:
- Connect Label Studio to a running **Triton Inference Server** instance
- Use the **MeDAL dataset** (from Hugging Face) — a collection of medical abstracts from PubMed
- Automatically generate **5 questions per abstract** in Label Studio using the LLM backend
- Generate corresponding **answers** in Label Studio
- Preprocess the dataset into **JSONL** format

---

### 📘 [`04-nemo-medal-qa-peft.ipynb`](04-nemo-medal-qa-peft.ipynb)

**Goal:**  
Fine-tune the **Llama 3.1 8B Instruct** model using the synthetic Q&A dataset generated from the MeDAL corpus.

Highlights:
- Reuse the **PEFT setup** from [`01-nemo-medchat-qa-peft.ipynb`](01-nemo-medchat-qa-peft.ipynb)
- Load and prepare the synthetic Q&A dataset
- Fine-tune the model using **LoRA**
- Save the resulting adapter weights for further evaluation or deployment

---

### 🧪 Exercise Proposal

**Task:** 
Deploy the fine-tuned model for the medical domain with Triton.

Follow [`02-triton-trtllm-model-deployment.ipynb`](02-triton-trtllm-model-deployment.ipynb) as a base to deploy the fine-tuned model from [`04-nemo-medal-qa-peft.ipynb`](04-nemo-medal-qa-peft.ipynb):
- Load the LoRA checkpoint during inference
- Export the model in TensorRT-LLM format
- Deploy the model using Triton Inference Server
- Configure 1 to 4 GPU replicas
- Validate inference across all replicas
- Check how the throughput scales with the number of replicas

---

## 🚀 Quick Start

To run the pipeline, follow the sequence:
1. Prepare the model and environment with [`01-medchat-qa-peft.ipynb`](01-medchat-qa-peft.ipynb)
2. Deploy an inference server with [`02-triton-trtllm-model-deployment.ipynb`](02-triton-trtllm-model-deployment.ipynb)
3. Create a synthetic medical Q&A dataset with [`03-ls-create-medal-qa-dataset.ipynb`](03-ls-create-medal-qa-dataset.ipynb)
4. Fine-tune the model on the synthetic data with [`04-nemo-medal-qa-peft.ipynb`](04-nemo-medal-qa-peft.ipynb)

---

## 🛠️ Requirements

- UCloud project environment with access to GPU resources
- [NVIDIA NeMo Framework](https://docs.cloud.sdu.dk/Apps/nemo.html)
- [Label Studio](https://docs.cloud.sdu.dk/Apps/label-studio.html)
- [NVIDIA Triton Inference Server](https://docs.cloud.sdu.dk/Apps/triton.html)
- Hugging Face Datasets: 
    - [MedChat-QA](https://huggingface.co/datasets/ngram/medchat-qa)
    - [MeDAL](https://huggingface.co/datasets/McGill-NLP/medal)

---

## 📜 License

This project is open-sourced under the [Apache 2.0 License](LICENSE).

