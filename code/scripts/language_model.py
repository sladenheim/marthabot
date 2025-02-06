from transformers import AutoTokenizer, AutoModelForCausalLM

def create_tokenizer(model, cache, token):
    tokenizer = AutoTokenizer.from_pretrained(model, 
                                              cache_dir = cache,
                                              access_token = token)
    return tokenizer

def create_model(model, cache, token):
    model = AutoModelForCausalLM.from_pretrained(model,
                                                 cache_dir = cache,
                                                 access_token = token,
                                                 device_map="auto",)
    return model

def load_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return tokenizer

def load_model(checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, 
                                                 device_map="auto")
    return model