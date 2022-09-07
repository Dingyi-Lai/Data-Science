import torch

from deen_mt_model import Translator

# oad models with and without attention
translator: Translator = torch.load(f"experiments/best-translation-model-attention-1024.pt", map_location='cpu')
print(translator)

text = "Ich mag Deep Learning !"

print("---- WITH ATTENTION ----")
for i in range(4):
    translator.generate_beam_search_translations_park(f"{text}\t", i+1)



