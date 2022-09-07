import time
import torch
import random
import torch.optim as optim

from deen_mt_helper_functions import plot_loss_curves, make_dictionary
from deen_mt_model import Translator

# Change parameters remotely
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=25,
                    help="number of epochs")
parser.add_argument("--mini_batch_size", type=int, default=8,
                    help="mini_batch_size")
parser.add_argument("--cuda", type=str, default='cpu',
                    help="which cuda device to use, -1 is cpu")
parser.add_argument("--character_level", action="store_true",
                    help="character level training")
parser.add_argument("--unk_threshold", type=int, default=0,
                    help="threshold how often a token needs to appear to not be masked by UNK")
parser.add_argument("--lr", type=float, default=0.1,
                    help="learning rate")
parser.add_argument("--rnn_hidden_size", type=int, default=64,
                    help="rnn_hidden_size")
parser.add_argument("--embed_size", type=int, default=10,
                    help="size/dimension of the embedding output")
parser.add_argument("--dataset", type=str, default="data/translations_5k.txt",
                    # choices=DATASETS,
                    help="Which dataset to train on")
parser.add_argument("--model_name", type=str, default=None,
                    help="Name of model within experiments folder")

args = parser.parse_args()


torch.manual_seed(1)

device = args.cuda

# more interesting text
input_filename = args.dataset

# All hyperparameters
learning_rate = args.lr
number_of_epochs = args.epochs
rnn_hidden_size = args.rnn_hidden_size
mini_batch_size = args.mini_batch_size
unk_threshold = args.unk_threshold
character_level = args.character_level

# device = 'cpu'

# more interesting text
input_filename = "data/translations_5k.txt"

# All hyperparameters
learning_rate = 0.1
number_of_epochs = 2
rnn_hidden_size = 64
mini_batch_size = 1
unk_threshold = 5
character_level = True
model_save_name = f"experiments/best-translation-model-attention.pt"

print(f"Training language model with \n - rnn_hidden_size: {rnn_hidden_size}\n - learning rate: {learning_rate}"
      f" \n - max_epochs: {number_of_epochs} \n - mini_batch_size: {mini_batch_size} \n - unk_threshold: {unk_threshold}")

# -- Step 1: Get a small amount of training data
training_data = []
with open(input_filename) as text_file:
    for line in text_file.readlines():

        source = line.split("\t")[0]
        target = line.split("\t")[1]

        if character_level:
            training_data.append(([char for char in source], [char for char in target]))
        else:
            # default is loading the corpus as sequence of words
            training_data.append(line.lower().strip().split(" "))

corpus_size = len(training_data)

training_data = training_data[:-round(corpus_size / 5)]
validation_data = training_data[-round(corpus_size / 5):-round(corpus_size / 10)]
test_data = training_data[-round(corpus_size / 10):]

print(
    f"\nTraining corpus has {len(training_data)} train, {len(validation_data)} validation and {len(test_data)} test sentences")

all_source_sentences = [pair[0] for pair in training_data]
all_target_sentences = [pair[1] for pair in training_data]

source_dictionary = make_dictionary(all_source_sentences, unk_threshold=unk_threshold, translation=True)
target_dictionary = make_dictionary(all_target_sentences, unk_threshold=unk_threshold, translation=True)

# initialize translator and send to device
model = Translator(source_vocabulary=source_dictionary,
                   target_vocabulary=target_dictionary,
                   rnn_size=rnn_hidden_size,
                   keep_context=True,
                   num_layers=2,
                   )
model.to(device)
print(model)

# --- Do a training loop

# define a simple SGD optimizer with a learning rate of 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# log the losses and accuracies
train_loss_per_epoch = []
validation_loss_per_epoch = []
validation_perplexity_per_epoch = []

# remember the best model
best_model = None
best_epoch = 0
best_validation_perplexity = 100000.

# Go over the training dataset multiple times
for epoch in range(number_of_epochs):

    print(f"\n - Epoch {epoch}")

    start = time.time()

    # shuffle training data at each epoch
    random.shuffle(training_data)

    train_loss = 0.

    import more_itertools

    for batch in more_itertools.chunked(training_data, mini_batch_size):
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Run our forward pass.
        loss = model.forward_loss(batch)

        # remember loss and backpropagate
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(training_data)

    # Evaluate and print accuracy at end of each epoch
    validation_perplexity, validation_loss = model.evaluate(validation_data)

    # remember best model:
    if validation_perplexity < best_validation_perplexity:
        print(f"new best model found!")
        best_epoch = epoch
        best_validation_perplexity = validation_perplexity

        # always save best model
        torch.save(model, model_save_name)

    # print losses
    print(f"training loss: {train_loss}")
    print(f"validation loss: {validation_loss}")
    print(f"validation perplexity: {validation_perplexity}")

    # append to lists for later plots
    train_loss_per_epoch.append(train_loss)
    validation_loss_per_epoch.append(validation_loss)
    validation_perplexity_per_epoch.append(validation_perplexity)

    end = time.time()
    print(f'{round(end - start, 3)} seconds for this epoch')

# do final test:
# load best model and do final test
best_model = torch.load(model_save_name)
test_accuracy = best_model.evaluate(test_data)

# print final score
print("\n -- Training Done --")
print(f" - using model from epoch {best_epoch} for final evaluation")
print(f" - final score: {test_accuracy}")

# make plots
plot_loss_curves(train_loss_per_epoch,
                 validation_loss_per_epoch,
                 validation_perplexity_per_epoch,
                 approach_name="Translation Model",
                 validation_label='Validation perplexity',
                 hyperparams={"rnn_hidden_size": rnn_hidden_size,
                              "lr": learning_rate})
