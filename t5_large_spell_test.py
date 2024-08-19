
from transformers import T5ForConditionalGeneration, AutoTokenizer

path_to_model = "T5-large-spell"

model = T5ForConditionalGeneration.from_pretrained("T5-large-spell")
tokenizer = AutoTokenizer.from_pretrained("T5-large-spell")
prefix = "grammar: "

sentence = "Th festeivаl was excelzecnt in many ways, and in particular it beinganinternational festjival sss a chаllenging, bet brilli an t ea."
sentence = prefix + sentence
import time
start = time.time()
encodings = tokenizer(sentence, return_tensors="pt")
generated_tokens = model.generate(**encodings, max_length=100, early_stopping=True)
answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(time.time()-start)
print(answer)

# ["If you bought something gorgeous, you will be very happy."]