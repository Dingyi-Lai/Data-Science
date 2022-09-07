import math
from typing import Dict
import random
import torch
import matplotlib.pyplot as plt
from textwrap import wrap
from transformers import BertTokenizer

from torch.nn.utils.rnn import pack_padded_sequence

def search(list, art):
    for i in range(len(list)):
        if list[i] == art:
            return list[i]
    return 'O'

def read_or_build_data_from_file(Path, data_set, build_training, subtoken, n, articles, corrector):
    if build_training:
        if subtoken:
            tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        tags = [0]*(int(n*1000/2)) + [1]*(int(n*1000/2))
        random.shuffle(tags)
        training_data = []
        store = []
        label = []
        j = 0
        with open(data_set) as text_file:

            for line1 in text_file.readlines():
                line = line1.strip()
                if subtoken:
                    source = tokenizer.tokenize(line)
                else:
                    source = line.split(" ")
                tag = []

                for i in range(len(source)):
                    is_lower = source[i].islower() # check whether lowercase

                    tag_n = search(articles, source[i].lower())
                    if (tag_n != 'O') & (tags[j] == 1):
                        articles_rest = list(filter(lambda num: num != source[i].lower(), articles))
                        if corrector:
                            tag_n = source[i]
                        else:
                            tag_n = 'ERR'
                        if is_lower:
                            source[i] = random.choice(articles_rest)
                        else:
                            source[i] = random.choice(articles_rest).title()
                    else:
                        tag_n = 'O'

                    tag.append(tag_n)

                seperator = '\t'
                store.append(seperator.join([' '.join(source), ' '.join(tag)]))
                training_data.append((source, tag))

                if j == (n*1000-1):
                    break
                else:
                    j = j + 1
        if subtoken:
            if corrector:
                store_name = 'training_data_'+str(n)+'k'+'_corr_sub.txt'
            else:
                store_name = 'training_data_'+str(n)+'k'+'_sub.txt'
        else:
            if corrector:
                store_name = 'training_data_'+str(n)+'k'+'_corr.txt'
            else:
                store_name = 'training_data_'+str(n)+'k.txt'

        with open(Path('data') / store_name, 'w') as f:  # save
            for line2 in store:
                f.write(line2)
                f.write('\n')
    else:
        training_data = []
        with open(data_set) as text_file:
            for line in text_file.readlines():
                sentence = line.split("\t")[0]
                tags = line.split("\t")[1]
                # each training data item is a tuple consisting of a sentence (list of words) and a list of labels
                training_data.append((sentence.strip().split(" "), tags.strip().split(" ")))
    return training_data

# make word dictionary from pre-trained word embeddings
def make_word_dictionary_from_pretrained(embedding_model):
    word_dictionary = {}
    for index, word in enumerate(embedding_model.index2word):
        word_dictionary[word] = index
    return word_dictionary


def make_dictionary(data, unk_threshold: int = 0, translation: bool = False):
    '''
    Makes a dictionary of words given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :param translation: whether it's for translation task
    :return: A dictionary of string keys and index values
    '''

    # First count the frequency of each distinct ngram
    word_frequencies = {}
    for sent in data:
        for word in sent:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1

    # Assign indices to each distinct ngram
    if translation:
        word_to_ix = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<STOP>': 3}
    else:
        word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix)

    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix


def make_label_dictionary(data) -> Dict[str, int]:
    '''
    Make a dictionary of labels.
    :param data: List of (sentence, label) tuples
    :return: A dictionary of string keys and index values
    '''
    word_frequencies = {}
    for sent in data:
        for word in sent:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1
    label_to_ix = {}
    for word, freq in word_frequencies.items():
        label_to_ix[word] = len(label_to_ix)

    return label_to_ix


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_label_vector(labels, label_to_ix):
    return torch.LongTensor([label_to_ix[label] for label in labels])


def make_decoder_input_vectors(sentences, word_to_ix):

    encoded = make_encoder_input_vectors(sentences, word_to_ix)
    target_inputs = encoded[:, :-1]
    target_targets = encoded[:, 1:]

    return target_inputs, target_targets


def make_encoder_input_vectors(sentences, word_to_ix):
    onehot_mini_batch = []

    lengths = [len(sentence) for sentence in sentences]
    longest_sequence_in_batch = max(lengths)

    for sentence in sentences:

        onehot_for_sentence = [word_to_ix["<START>"]]

        # move a window over the text
        for word in sentence:

            # look up ngram index in dictionary
            if word in word_to_ix:
                onehot_for_sentence.append(word_to_ix[word])
            else:
                onehot_for_sentence.append(word_to_ix["<UNK>"] if "<UNK>" in word_to_ix else 0)

        # append a STOP index
        onehot_for_sentence.append(word_to_ix["<STOP>"])

        # fill the rest with PAD indices
        for i in range(longest_sequence_in_batch - len(sentence)):
            onehot_for_sentence.append(word_to_ix["<PAD>"])

        onehot_mini_batch.append(onehot_for_sentence)

    onehot_mini_batch = torch.tensor(onehot_mini_batch)

    return onehot_mini_batch


def make_onehot_vectors_for_language_modeling(sentences, word_to_ix):
    onehot_mini_batch = []

    lengths = [len(sentence) for sentence in sentences]
    longest_sequence_in_batch = max(lengths)

    for sentence in sentences:

        onehot_for_sentence = [word_to_ix["<START>"]]

        # move a window over the text
        for word in sentence:

            # look up ngram index in dictionary
            if word in word_to_ix:
                onehot_for_sentence.append(word_to_ix[word])
            else:
                onehot_for_sentence.append(word_to_ix["<UNK>"] if "<UNK>" in word_to_ix else 0)

        # append a STOP index
        onehot_for_sentence.append(word_to_ix["<STOP>"])

        # fill the rest with PAD indices
        for i in range(longest_sequence_in_batch - len(sentence)):
            onehot_for_sentence.append(word_to_ix["<PAD>"])

        onehot_mini_batch.append(onehot_for_sentence)

    onehot_mini_batch = torch.tensor(onehot_mini_batch)

    inputs = onehot_mini_batch[:, :-1]
    targets = onehot_mini_batch[:, 1:]

    return inputs, targets


def make_onehot_vectors(sentences, word_to_ix):
    onehot_mini_batch = []

    longest_sequence_in_batch = max([len(sentence) for sentence in sentences])

    for sentence in sentences:

        onehot_for_sentence = []

        # move a window over the text
        for word in sentence:

            # look up ngram index in dictionary
            if word in word_to_ix:
                onehot_for_sentence.append(word_to_ix[word])
            else:
                onehot_for_sentence.append(word_to_ix["UNK"] if "UNK" in word_to_ix else 0)

        for i in range(longest_sequence_in_batch - len(sentence)):
            onehot_for_sentence.append(0)

        onehot_mini_batch.append(onehot_for_sentence)

    return torch.tensor(onehot_mini_batch)


def plot_loss_curves(loss_train,
                     loss_val,
                     accuracy_val,
                     approach_name: str,
                     hyperparams,
                     validation_label='Validation accuracy'):
    last_finished_epoch = len(loss_train)
    epochs = range(1, last_finished_epoch + 1)
    hyperparam_pairs = [f"{key}{hyperparams[key]}" for key in hyperparams]

    file_name = f"experiments/loss-curves-{approach_name}-" + "-".join(hyperparam_pairs).replace("/", "-") + ".png"
    title_text = ", ".join([f"{key}:{hyperparams[key]}" for key in hyperparams])

    fig, ax1 = plt.subplots()

    color = 'g'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_train, 'r', label='Training loss')
    ax1.plot(epochs, loss_val, 'g', label='Validation loss')
    ax1.tick_params(axis='y', labelcolor=color)
    title = ax1.set_title("\n".join(wrap(title_text, 60)))
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'k'  # k := black
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, accuracy_val, 'black', label=validation_label)
    ax2.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.xticks(range(5, math.floor((last_finished_epoch + 1) / 5) * 5, 5))
    plt.savefig(file_name)
    plt.show()
