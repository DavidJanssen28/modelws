from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class LLMInterface:
    def __init__(self, model_name="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    def generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
