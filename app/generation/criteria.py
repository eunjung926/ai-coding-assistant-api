from transformers import StoppingCriteria


class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length: int, eof_strings: list[str], tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        return all(
            any(stop_string in generation for stop_string in self.eof_strings)
            for generation in decoded_generations
        )
