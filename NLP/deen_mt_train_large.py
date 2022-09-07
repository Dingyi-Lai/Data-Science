from pathlib import Path
import torch


# Task 2 is:
# - to use a subword-level model (BPE or sentence piece)
# - train a larger model on GPU (512 states, 50 epochs, mbz 32, at least 20 epochs) on 15k translations
# - save a model with 512 states as 512-translations-model.pt and commit to repo
# - add a temperature parameter to "translate" function so that the below script executes
is_attention = False
# Task 1 is:
# - to implement a sequence-to-sequence model
# - train it on the translations_sanity_check.txt data.
# - save a model with 128 states as sanity-translations-model.pt and commit to repo
# - write a "translate" function so that the model passes the below assertions


model_path = Path('experiments') / 'best-translation-model-attention-1024.pt'


# load the model again
translator = torch.load(model_path, map_location=torch.device('cpu'))
print(translator)

source = "Wir sehen uns ."
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"{source} is translated as {translator.generate_translations(source, temperature)}")

source = "Wir sind echt gut ."
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"{source} is translated as {translator.generate_translations(source, temperature)}")

source = "Wir verstehen das ."
for temperature in [0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"{source} is translated as {translator.generate_translations(source, temperature)}")