from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Generator:

  def __init__(self):
    model_name = "Qwen/Qwen2-7B-Instruct" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", force_download=True, low_cpu_mem_usage=True) # device_map to handle cpu/gpu

  def generate(self, context, question):

    #prompt = f"""
    #  You are a scientific assistant.

    #  Answer the question using ONLY the context.

    #  If the answer is not in the context, say "I don't know".

    #  Context:
    #  {context}

    #  Question:
    #  {question}

    #  Answer:
    #  """

    prompt = f"""<|im_start|>system
    You are a scientific assistant.
    
    RULES:
    - Use ONLY the provided abstracts.
    - Do NOT use prior knowledge.
    - If the answer is not explicitly in the abstracts, say: "Not found in the provided papers."
    - Do NOT guess or infer.
    
    <|im_end|>
    <|im_start|>user
    Abstracts:
    {context}
    
    Question: {question}
    
    Return only information supported by the abstracts.
    <|im_end|>
    <|im_start|>assistant
    """

    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) #3 return_tensors in PyTorch

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=100,
        eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=self.tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.5 
    )

    input_length = inputs["input_ids"].shape[1]
    result = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    result = result.split("I don't know")[0].strip()
    return result