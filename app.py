import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "microsoft/phi-2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, "lora_adapter")


def chat_fn(message, history):
    prompt = f"""
	You are a Linux troubleshooting expert.

	Rules:
	- Always give command-based answers
	- Keep answers short and practical
	- Prefer terminal commands over explanations
	- If possible, include example usage

	### Instruction:
	{message}

	### Response:
	"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()

    return response


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 Linux Troubleshooting Assistant  
        Fine-tuned Phi-2 using LoRA  
        """
    )

    chatbot = gr.ChatInterface(
        fn=chat_fn,
        textbox=gr.Textbox(placeholder="Ask something about Linux...", container=False),
    )

demo.launch()
