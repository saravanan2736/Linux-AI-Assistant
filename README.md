# 🐧 Linux AI Assistant

A domain-specific chatbot built by fine-tuning a Large Language Model on limited hardware using LoRA.

---

## 🚀 Features

- Fine-tuned Phi-2 using LoRA (Parameter Efficient Fine-Tuning)
- Runs locally on laptop GPU
- Linux troubleshooting assistant
- Interactive chat interface (Gradio)
- Command-based responses

---

## 🧠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Gradio

---

## ⚡ How It Works

1. Dataset preparation (instruction format)
2. LoRA fine-tuning on Phi-2
3. Prompt conditioning for Linux specialization
4. Deployment using Gradio UI

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
python app.py
