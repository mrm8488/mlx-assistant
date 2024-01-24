from mlx_lm import load, generate


class TextGen:
    def __init__(self, model_id="microsoft/phi-2"):
        model, tokenizer = load("mistralai/Mistral-7B-v0.1")
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, verbose=True):
        response = generate(self.model, prompt=prompt, verbose=verbose)
        return response["choices"][0]["text"]
