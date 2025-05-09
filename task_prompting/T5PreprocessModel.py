from transformers import PreTrainedModel, T5ForConditionalGeneration, T5Tokenizer


class T5PreprocessModel:
    def __init__(self, w_path):
        self.model = T5ForConditionalGeneration.from_pretrained(w_path)
        self.model.to('cuda')
        self.tokenizer = T5Tokenizer.from_pretrained(w_path)
        special_tokens_dict = {
            'additional_special_tokens': ['<check_if_negated>', '<original>', '<variation>', '<canonize>']}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))


    def get_output(self, input_text):
        inputs = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].to('cuda')
        outputs = self.model.generate(input_ids)
        out_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if '<canonize>' in input_text:
            return out_str
        return int(out_str == 'Yes')