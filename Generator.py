from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class Generator:

  def __init__(self):
    model_name = "Qwen/Qwen2-7B-Instruct" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", force_download=False, low_cpu_mem_usage=True) # device_map to handle cpu/gpu

  def clean_citations(self, text, valid_ids):

    text = re.sub(r'\\textit\{arXiv:\[([^\]]+)\]\}', r'[\1]', text)
    text = re.sub(r'arXiv:\[([^\]]+)\]', r'[\1]', text)
    
    def replace(match):
        cit = match.group(1).strip()
        if cit in valid_ids:
            return f"[{cit}]"
        return ""
    
    return re.sub(r'\[([^\]]+)\]', replace, text)
    
  def generate(self, context, question, allowed_ids):

    prompt = f"""<|im_start|>system
    You are an information extraction system. Answer ONLY from the provided abstracts.
    
    RULES:
    - Maximum 3 bullet points
    - Each bullet MUST end with exactly one arxiv ID in brackets
    - Valid IDs: {', '.join(allowed_ids)}
    - Use ONLY these IDs, no others
    - If not found in abstracts say: "Not found in the provided papers."
    
    FORMAT EXAMPLE:
    * Transformers dominate NLP tasks [2311.17633]
    * They enable source code processing [2010.07987]
    <|im_end|>
    
    <|im_start|>user
    Context:
    {context}
    
    Question:
    {question}
    <|im_end|>
    
    <|im_start|>assistant
    *"""

    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) #3 return_tensors in PyTorch

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=250,
        eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=self.tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2 
    )

    input_length = inputs["input_ids"].shape[1]
    result = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    result = result.split("I don't know")[0].strip()

    result = self.clean_citations(result, allowed_ids)
    return result