import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    # Source for the Encoder: https://github.com/udacity/CVND---Image-Captioning-Project/tree/master
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)
        self.projection = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        # self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, features, captions):
        batch_size, embed_size = features.shape
        assert embed_size == self.embed_size
        
        batch_size2, captions_len = captions.shape
        assert batch_size2 == batch_size

        # get the sequence by taking the "captions" minus the last word. Start token has been added already
        input_seq = captions[:, :-1]  # shape (batch_size, captions_len - 1)
        assert input_seq.shape == (batch_size, captions_len - 1)
        
        # pass through the Embedding layer
        word_embeddings = self.word_embedding_layer(input_seq)  # shape (batch_size, captions_len - 1, embed_size)
        assert word_embeddings.shape == (batch_size, captions_len - 1, self.embed_size)
        
        resized_features = features.unsqueeze(1)  # shape (batch_size, 1, embed_size)
        assert resized_features.shape == (batch_size, 1, self.embed_size)
        input_seq_concat = torch.cat((resized_features, word_embeddings, ), dim=1)
        
        # then pass the sequence and the hidden, cell tuple into LSTM
        lstm_outputs = self.lstm(input_seq_concat)  # shape (batch_size, captions_len, hidden_size)
        assert lstm_outputs[0].shape == (batch_size, captions_len, self.hidden_size)
        
        # then pass through a linear projection. What's the resulting shape? Assert
        projections = self.projection(lstm_outputs[0])  # shape (batch_size, captions_len, vocab_size)
        assert projections.shape == (batch_size, captions_len, self.vocab_size)
        
        # then pass through a log softmax? What are the shapes? 
        # NOT NECESSARY, since nn.CrossEntropy expects non-normalized logits
        # outputs = self.softmax(projections)
        scores = projections
        assert scores.shape == (batch_size, captions_len, self.vocab_size)
        
        # then output
        return scores


    def sample(self, inputs, states=None, max_len=20, beam_width=5, sos_token=0, eos_token=1, output_multiple=False):
        """
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        beam_width: the number of candidates to keep and try to expand
        max_length: the maximum length of generated sequences (excluding SOS and EOS)
        SOS_token: default is 0
        EOS_token: default is 1
        """
        self.eval()
        # inputs: features (shape must be (batch_size, embed_size)) - with batch size of 1
        batch_size, embed_size = inputs.shape
        assert (batch_size == 1) and (embed_size == self.embed_size)

        # initializing: "candidates": a list with "beam_width" sequences of one token - the SOS token
        # each sequence is actually a tuple: (tensor of tokens, log probability)
        current_device = next(self.parameters()).device
        candidates = [(torch.LongTensor([sos_token]).to(current_device), 0) ]

        while (True):
            still_expanding = False
            new_candidates = []
            # for each candidate in candidate list ("candidates")
            for candidate in candidates:
                
                # if already included EOS, then just put that candidate directly into new_candidates
                current_seq = candidate[0]
                current_seq_log_prob_score = candidate[1]
                if current_seq[-1].item() == eos_token:
                    new_candidates.append(candidate)

                # if not yet included EOS: 
                else:
                    still_expanding = True
                    # need to format the candidate in the form to pass into the decoder
                    # each candidate should have be converted to (batch_size, length_so_far) shape with batch_size of 1 here
                    # also need to pad the token with one trailing EOS token (see the forward function of the decoder)
                    current_seq_feed_decoder = current_seq.unsqueeze(0)
                    current_seq_feed_decoder = torch.cat((current_seq_feed_decoder, torch.LongTensor([[eos_token]]).to(current_device)), dim=1)
                    
                    # pass the features and the candidate sequence through the decoder,
                    decoder_logit_outputs = self.forward(inputs, current_seq_feed_decoder)
                    
                    # convert logits into log-probability
                    decoder_outputs = torch.nn.functional.log_softmax(decoder_logit_outputs, dim=-1)
                    
                    # assert decoder_outputs.shape == ()
                    assert decoder_outputs.shape == (1, len(current_seq) + 1, self.vocab_size)
                    # (if the candidate already reached a maximum length - 1, then just force-append the EOS token, compute the score and add this into the new_candidates list
                    if len(current_seq) - 1 == max_len:  # sequence length should not count the beginning sos_token and the trailing eos_token
                        # force this sequence to end by accepting the eos token
                        new_seq = torch.cat((current_seq, torch.LongTensor([eos_token]).to(current_device)), dim=0)
                        new_seq_log_prob = current_seq_log_prob_score + decoder_outputs[0, -1, eos_token]
                        # new_seq_log_prob /= (len(new_seq)**0.7)
                        new_candidate = (new_seq, new_seq_log_prob)
                        new_candidates.append(new_candidate)
                    
                    else:
                        # examine the outputs (probability of each possible next word in the vocabulary)
                        # form new_candidate by appending each next word in vocabulary, 
                        # adding the log prob to get the new score,
                        # then put this new_candidate into new_candidates
                        for next_token in range(self.vocab_size):  # TODO check how other solutions access the tokens in the vocab
                            new_seq = torch.cat((current_seq, torch.LongTensor([next_token]).to(current_device)), dim=0)
                            new_seq_log_prob = current_seq_log_prob_score + decoder_outputs[0, -1, next_token]
                            # new_seq_log_prob /= (len(new_seq)**0.7)
                            new_candidate = (new_seq, new_seq_log_prob)
                            new_candidates.append(new_candidate)
                    

            # if no candidate has just been expanded, then could exit
            if not still_expanding:
                break
                
            # update "candidates" (take the current top crop in new_candidates)
            new_candidates_ranked = sorted(new_candidates, key=lambda candidate: candidate[1], reverse=True)
            candidates = new_candidates_ranked[:beam_width]
            # then repeat
        
        
        # outputs
        if output_multiple:
            return candidates
        else:
            return [token.item() for token in candidates[0][0]]
        
         
    def sample2(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        batch_size, embed_size = inputs.shape
        assert (batch_size == 1) and (embed_size == self.embed_size)
        inputs = inputs.unsqueeze(1)
        for index in range(max_len):
            
            
            lstm_out, states = self.lstm(inputs, states)

            
            lstm_out = lstm_out.squeeze(1)
            outputs = self.projection(lstm_out)
            
            
            target = outputs.max(1)[1]
            
            
            predicted_sentence.append(target.item())
            
            
            inputs = self.word_embedding_layer(target).unsqueeze(1)
            
        return predicted_sentence
        
        # if states == None:
        #     states = (torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device),
        #               torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device))
        outputs = list()
        for i in range(max_len):
            scores, states = self.lstm(inputs, states)  # scores: (1, 1, vocab_size)
            scores = self.projection(scores.squeeze(1))  # (1, vocab_size)
            output = scores.max(1)[1]
            outputs.append(output.item())
            inputs = self.word_embedding_layer(output.unsqueeze(1))  # (1, 1, embed_size)
        return outputs