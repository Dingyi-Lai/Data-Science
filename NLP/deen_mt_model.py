import heapq
import math
import torch.nn.functional
import torch
import numpy
from deen_mt_helper_functions import make_encoder_input_vectors, make_decoder_input_vectors

class Translator(torch.nn.Module):

    def __init__(self,
                 source_vocabulary,
                 target_vocabulary,
                 rnn_size: int,
                 source_embedding_size: int = 50,
                 target_embedding_size: int = 50,
                 keep_context: bool = False,
                 num_layers: int = 1,
                 # use_attention: bool = True,
                 ):
        super().__init__()

        # the two dictionaries
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

        # the two encoder layers
        self.source_embedding = torch.nn.Embedding(len(source_vocabulary), source_embedding_size)
        self.source_lstm = torch.nn.LSTM(source_embedding_size, rnn_size, batch_first=True, num_layers=num_layers)

        # the three decoder layers
        self.target_embedding = torch.nn.Embedding(len(target_vocabulary), target_embedding_size)

        self.target_lstm = torch.nn.LSTM(target_embedding_size,
                                         rnn_size,
                                         batch_first=True)
        # without multilayer
        # in sequence labeling and neural language modeling, I often just use 1 layer RNNs

        self.hidden2tag = torch.nn.Linear(rnn_size * 2 if keep_context else rnn_size, len(target_vocabulary))

        self.keep_context = keep_context

    def forward_loss(self, batch):

        # get the device this model is on
        device = next(self.parameters()).device

        source_sentences = [pair[0] for pair in batch]
        target_sentences = [pair[1] for pair in batch]

        # make datapoint and its label into a vector respectively
        source_inputs = make_encoder_input_vectors(source_sentences, self.source_vocabulary).to(device)

        target_inputs, target_targets = make_decoder_input_vectors(target_sentences, self.target_vocabulary)
        target_inputs = target_inputs.to(device)
        target_targets = target_targets.to(device)

        # encode source input
        encoder_outputs, encoder_last_hidden = self.forward_encode(source_inputs)

        # print(encoder_last_hidden)
        # only use last hidden state of top layer
        # debug1 = encoder_last_hidden[0][-1,:,:].unsqueeze(0)
        encoder_last_hidden = (encoder_last_hidden[0][-1,:,:].unsqueeze(0), encoder_last_hidden[1][-1,:,:].unsqueeze(0))

        # encode target inputs, but begin with last hidden state of encoder
        log_probabilities_for_each_class, decoder_last_hidden = self.forward_decode(target_inputs, encoder_last_hidden,
                                                                                    encoder_outputs)

        # explanation: the loss function can't handle mini-batches, so we flatten all predictions and targets for
        # the whole mini-batch into one long list
        flattened_log_probabilities_for_each_class = log_probabilities_for_each_class.flatten(end_dim=1)
        flattened_targets = target_targets.flatten()

        # compute the loss
        loss = torch.nn.functional.nll_loss(flattened_log_probabilities_for_each_class, flattened_targets)

        return loss

    def forward_encode(self, source):

        # encode source input
        embedded_source = self.source_embedding(source)

        outputs, hidden = self.source_lstm(embedded_source)

        return outputs, hidden

    def forward_decode(self, target, hidden, encoder_outputs=None):

        # embed target text
        embedded_target = self.target_embedding(target)

        all_predictions = []
        for i in range(target.size(1)):
            input = embedded_target[:, i, :].unsqueeze(0)

            # use last state as input to target
            decoder_rnn_output, hidden = self.target_lstm(input, hidden)

            if self.keep_context:

                # multiply all encoder outputs with current output of decoder to get all attention scores
                attention_scores = torch.matmul(encoder_outputs.squeeze(), decoder_rnn_output.squeeze())

                # softmax attention scores to get distribution
                attention_distribution = torch.nn.functional.softmax(attention_scores, dim=0)

                # another multiplication to get distribution
                attention_output = torch.matmul(attention_distribution, encoder_outputs.squeeze()).unsqueeze(0).unsqueeze(0)

                # concat attention output to decoder rnn output and send through linear layer
                logits = self.hidden2tag(torch.cat((decoder_rnn_output, attention_output), 2))

            else:
                logits = self.hidden2tag(decoder_rnn_output)

            prediction = torch.nn.functional.log_softmax(logits, dim=2)
            all_predictions.append(prediction)

        prediction = torch.cat(all_predictions, dim=1)

        return prediction, hidden

    # TODO: implement this methdd
    def generate_beam_search_translations(self, text_to_translate, beam_size: int):

        device = next(self.parameters()).device
        inv_map = {v: k for k, v in self.target_vocabulary.items()}

        inputs = make_encoder_input_vectors([text_to_translate], self.source_vocabulary).to(device)
        print(inputs)

        # encode source input
        encoder_outputs, hidden = self.forward_encode(inputs[0].unsqueeze(0))

        target_input = torch.tensor([[self.target_vocabulary['<START>']]], device=device)

        hypothesis = [[(target_input, hidden, 0)]]  ### first hypothesis is just <START>
        final_candidates = []

        for i in range(100):  ### token limit
            new_hypothesis = []
            for hypo in hypothesis:
                # embed target text
                output, hidden = self.forward_decode(hypo[-1][0], hypo[-1][1])
                ### predict next tokens from previous token and hidden state for each hypo.

                log_prob, word_idx = output.squeeze().cpu().topk(beam_size)  ### select top k tokens with best log_prob

                ### append top token, hidden state and log_prob to hypo.
                for alt in range(len(log_prob)):
                    if inv_map[word_idx[alt].item()] == '<STOP>':
                        final_candidates.append(hypo + [
                            (torch.tensor([[word_idx[alt].item()]], device=device), hidden, log_prob[alt].item())])
                    else:
                        new_hypothesis.append(hypo + [
                            (torch.tensor([[word_idx[alt].item()]], device=device), hidden, log_prob[alt].item())])

            ### scores von neuen hypothesis berechnen, ranken, topk auswählen um 'schlechte hypothesis zu eliminieren
            scores = []
            for nh in new_hypothesis:
                print(nh)
                score = 0
                for word in nh:
                    print(word)
                    score += word[2]  ### score is sum of all log_probs of the individual tokens
                scores.append(score)


            ### best scoring hypothesis indices
            best_hypo_idx = sorted(range(len(scores)), key=lambda i: scores[i])[-beam_size:]

            ### replace the old hypothesis with the best k new_hypothesis
            hypothesis = []
            for idx in best_hypo_idx:
                hypothesis.append(new_hypothesis[idx])

        ### store hypothesis that have not yet terminated in final candidates to rank them
        for hypo in hypothesis:
            final_candidates.append(hypo)

        print(len(final_candidates))

        scores = []
        for hypo in final_candidates:
            score = 0
            for token in hypo:
                score += token[2]
            scores.append(score / len(hypo))

        
        best_candidate = final_candidates[numpy.argmax(scores)]

        translation = ""
        for token in best_candidate[1:-1]:  ### omit <START> and <STOP> token
            translation += inv_map[token[0].item()]

        # print(f"'{''.join(text_to_translate)}' is translated as: {translation}")
        return translation
        # TODO: implement this methdd

    def generate_beam_search_translations_martinro(self, text_to_translate, beam_size: int = 1):

        class Beam:
            def __init__(self, target_vocabulary, hidden_state):
                self.target_vocabulary = target_vocabulary

                self.last_hidden_state = hidden_state
                self.indices = [self.target_vocabulary['<START>']]
                self.logits = [0]

            def likelihood(self):
                return sum(self.logits) / len(self.logits)

            def is_stop(self):
                return self.indices[-1] == self.target_vocabulary['<STOP>']

            def get_inputs(self, device):
                last_index = self.indices[-1]
                target_input = torch.tensor([[last_index]], device=device)
                return target_input, self.last_hidden_state

            def next(self, hidden_state, word_idx, logit):
                beam = Beam(self.target_vocabulary, hidden_state)
                beam.indices = [*self.indices, word_idx]
                beam.logits = [*self.logits, logit]
                return beam

            def text(self, inv_map, separator=''):
                chars = [inv_map[int(word_idx)] for word_idx in self.indices]
                if self.is_stop():
                    return separator.join(chars[1:-1])
                else:
                    return separator.join(chars[1:])

        if beam_size < 1:
            raise ValueError(f'Beam size must be greater than 0, not {beam_size}')

        device = next(self.parameters()).device
        inv_map = {v: k for k, v in self.target_vocabulary.items()}
        inputs = make_encoder_input_vectors([text_to_translate], self.source_vocabulary).to(device)

        # encode source input
        encoder_outputs, hidden = self.forward_encode(inputs[0].unsqueeze(0))

        beams = [Beam(self.target_vocabulary, hidden)]
        for _ in range(100):
            next_beams = []
            beams_stop = True
            for beam in beams:
                if not beam.is_stop():
                    beams_stop = False
                    output, hidden = self.forward_decode(*beam.get_inputs(device))

                    # embed target text
                    top_k_logits = output.squeeze().cpu().topk(beam_size)

                    for index, logit in zip(top_k_logits.indices, top_k_logits.values):
                        next_beams.append(beam.next(hidden, index, logit))
                else:
                    next_beams.append(beam)
            if beams_stop:
                break

            # sort by odds, highest first
            beams = sorted(next_beams, key=Beam.likelihood, reverse=True)[:beam_size]

        print(f"'{''.join(text_to_translate)}' is translated as: {beams[0].text(inv_map)}")

    def generate_beam_search_translations_park(self, text_to_translate: str, beam_size: int):
        device = next(self.parameters()).device
        inv_map = {v: k for k, v in self.target_vocabulary.items()}

        inputs = make_encoder_input_vectors([text_to_translate], self.source_vocabulary).to(device)

        # encode source input
        encoder_outputs, hidden = self.forward_encode(inputs[0].unsqueeze(0))

        # only use last hidden state of top layer
        hidden = (hidden[0][-1,:,:].unsqueeze(0), hidden[1][-1,:,:].unsqueeze(0))

        target_input = torch.tensor([[self.target_vocabulary['<START>']]], device=device)

        # The first run of beam search
        sequences = list()

        # do a forward decode and get top k predictions for the first symbol
        output, hidden = self.forward_decode(target_input, hidden, encoder_outputs)
        first_symbol_predictions = output.squeeze().cpu().detach().topk(beam_size)

        # create one candidate hypothesis for each prediction
        for j in range(beam_size):
            # each candidate is a tuple consisting of the predictions so far, the last hidden state and the log
            # probabilities
            prediction_index = first_symbol_predictions.indices[j].item()
            prediction_log_probability = first_symbol_predictions.values[j]
            candidate = [[prediction_index], hidden, prediction_log_probability]
            sequences.append(candidate)

        # variables needed for further beam search
        n_completed = 0
        final_candidates = list()

        # Beam search after the first run
        for i in range(100):
            new_sequences = list()

            # expand each current candidate
            for seq, hid, score in sequences:
                target_input = torch.tensor([[seq[-1]]], device=device)

                # do a forward decode and get top k predictions
                output, hidden = self.forward_decode(target_input, hid, encoder_outputs)
                top_k_predictions = output.squeeze().cpu().detach().topk(beam_size)

                # go through each of the top k predictions
                for j in range(beam_size):

                    prediction_index = top_k_predictions.indices[j].item()
                    prediction_log_probability = top_k_predictions.values[j]

                    # add their log probability to previous score
                    s = score + prediction_log_probability

                    # if this prediction is a STOP symbol, set it aside
                    if prediction_index == self.target_vocabulary['<STOP>']:
                        candidate = [seq, s]
                        final_candidates.append(candidate)
                        n_completed += 1
                    # else, create a new candidate hypothesis with updated score and prediction sequence
                    else:
                        candidate = [seq + [prediction_index], hidden, s]
                        new_sequences.append(candidate)

            # order final candidates by score (in descending order)
            seq_sorted = sorted(new_sequences, key=lambda tup: tup[2], reverse=True)

            # only use top k hypothesis as starting point for next iteration
            sequences = seq_sorted[:beam_size]

        # normalize scores by length
        for i in range(len(final_candidates)):
            final_candidates[i][1] = final_candidates[i][1] / len(final_candidates[i][0])

        # order final candidates by score (in descending order)
        ordered = sorted(final_candidates, key=lambda tup: tup[1], reverse=True)
        best_sequence = ordered[0]

        generated_words = list()
        for i in best_sequence[0]:
            generated_words.append(inv_map[i])

        separator = ''  # if self.is_character_level else ' '
        print(f"'{''.join(text_to_translate)}' is translated as: {separator.join(generated_words)}")

    def generate_translations(self, text_to_translate, temperature=1.0):

        device = next(self.parameters()).device
        inv_map = {v: k for k, v in self.target_vocabulary.items()}

        inputs = make_encoder_input_vectors([text_to_translate], self.source_vocabulary).to(device)

        # encode source input
        encoder_outputs, hidden = self.forward_encode(inputs[0].unsqueeze(0))

        target_input = torch.tensor([[self.target_vocabulary['<START>']]], device=device)

        generated_words = []
        for i in range(100):

            # embed target text
            output, hidden = self.forward_decode(target_input, hidden, encoder_outputs)

            word_idx = output.squeeze().div(temperature).exp().cpu().argmax(0)

            # word_weights = output[:, -1, :].squeeze().div(temperature).exp().cpu()
            # word_idx = torch.multinomial(word_weights, 1)[0]

            target_input.fill_(word_idx)

            if word_idx == self.target_vocabulary['<STOP>']: break

            generated_words.append(inv_map[word_idx.item()])

        separator = ''  # if self.is_character_level else ' '
        print(f"'{''.join(text_to_translate)}' is translated as: {separator.join(generated_words)}")

    def evaluate(self, test_data):

        source_sentences = [pair[0] for pair in test_data]

        with torch.no_grad():

            for i in range(5):
                self.generate_beam_search_translations_park(source_sentences[i], beam_size=4)

            aggregate_loss = 0.

            # go through all test data points
            for instance in test_data:
                aggregate_loss += self.forward_loss([instance])

            aggregate_loss = aggregate_loss / len(test_data)

            return math.exp(aggregate_loss), aggregate_loss
