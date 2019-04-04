import pretty_midi
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import os
import random
import sys
import matplotlib.pyplot as plt


def midi_filename_to_piano_roll(midi_filename):
    # return an array of shape T*128
    midi = pretty_midi.PrettyMIDI(midi_filename)
    midi.remove_invalid_notes()
    piano_roll = midi.get_piano_roll()
    # Binarize the pressed notes
    piano_roll[piano_roll > 0] = 1
    assert piano_roll.shape[0] == 128
    return piano_roll.transpose()


def pad_piano_roll(piano_roll, max_length):
    return np.pad(piano_roll, ((0, max_length-piano_roll.shape[0]), (0, 0)), 'constant')


def seq_length(sample):
    piano_roll = sample[0]
    return piano_roll.shape[0]


def custom_collate_fn_ordered(batch):
    batch.sort(key=seq_length, reverse=True)
    max_len = seq_length(batch[0])
    padded_input = torch.from_numpy(np.concatenate(
        [np.expand_dims(pad_piano_roll(piano_roll, max_len), axis=0) for piano_roll, _ in batch], axis=0))
    padded_output = torch.from_numpy(np.concatenate(
        [np.expand_dims(pad_piano_roll(piano_roll, max_len), axis=0) for _, piano_roll in batch], axis=0))
    lengths = [piano_roll.shape[0] for piano_roll, _ in batch]
    return padded_input.transpose(0,1).float(), padded_output.transpose(0,1).long(), lengths


class NotesGenerationDataset(data.Dataset):
    
    def __init__(self, midi_folder_path, longest_sequence_length=None):
        self.midi_full_filenames = []
        for root, directories, filenames in os.walk(midi_folder_path):
            for filename in filenames:
                self.midi_full_filenames.append(os.path.join(root, filename))
    
    def __len__(self):
        return len(self.midi_full_filenames)
    
    def __getitem__(self, index):
        midi_full_filename = self.midi_full_filenames[index]
        piano_roll = midi_filename_to_piano_roll(midi_full_filename)
        # Shifted by one time step
        input_sequence = piano_roll[:-1, :]
        ground_truth_sequence = piano_roll[1:, :]
        return (input_sequence, ground_truth_sequence)


class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes, n_layers=2):        
        super(RNN, self).__init__()
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.logits_fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_sequences, input_sequences_lengths, hidden=None):
        notes_encoded = self.notes_encoder(input_sequences)      
        # Here we run rnns only on non-padded regions of the batch
        packed = torch.nn.utils.rnn.pack_padded_sequence(notes_encoded, input_sequences_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        logits = self.logits_fc(outputs)  # T*B*128
        logits = logits.transpose(0, 1).contiguous()  # B*T*128
        neg_logits = (1 - logits)
        
        # Since the BCE loss doesn't support masking, we use the crossentropy
        binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()  # B*T*128*2
        logits_flatten = binary_logits.view(-1, 2)  # (B*T*128)*2

        return logits_flatten, hidden


def validation(valset_loader):
    full_val_loss = 0.0
    overall_sequence_length = 0.0
    for input_sequences_batch, output_sequences_batch, sequences_lengths in valset_loader:
        output_sequences_batch =  Variable( output_sequences_batch.contiguous().view(-1).cuda() )
        input_sequences_batch = Variable( input_sequences_batch.cuda() )
        logits, _ = rnn(input_sequences_batch, sequences_lengths)
        loss = criterion_val(logits, output_sequences_batch)
        full_val_loss += loss.item()
        overall_sequence_length += sum(sequences_lengths)
    full_val_loss /= (overall_sequence_length * 128)
    return full_val_loss


if __name__ == "__main__":
    print("Loading Dataset...")
    dataset = NotesGenerationDataset('/home/jlamare/Documents/CMU/10-615/Project3/Beats/')

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(0.20 * dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainset_loader = torch.utils.data.DataLoader(dataset, collate_fn=custom_collate_fn_ordered, batch_size=32, drop_last=True, sampler=train_sampler)
    valset_loader = torch.utils.data.DataLoader(dataset, collate_fn=custom_collate_fn_ordered, batch_size=32, drop_last=False, sampler=valid_sampler)
    print("Dataset loaded")

    rnn = RNN(input_size=128, hidden_size=512, num_classes=128).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_val = nn.CrossEntropyLoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(rnn.parameters())

    clip = 1.0
    epochs_number = 200
    loss_mean_list = []
    val_list = []
    best_val_loss = float("inf")

    for epoch_number in range(epochs_number):

        print("EPOCH ", epoch_number)
        loss_list = []

        for input_sequences_batch, output_sequences_batch, sequences_lengths in trainset_loader:
            output_sequences_batch =  Variable( output_sequences_batch.contiguous().view(-1).cuda() )
            input_sequences_batch_var = Variable( input_sequences_batch.cuda() )
            optimizer.zero_grad()
            logits, _ = rnn(input_sequences_batch_var, sequences_lengths)
            loss = criterion(logits, output_sequences_batch)
            loss_list.append( loss.item() )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
            optimizer.step()

        del input_sequences_batch, output_sequences_batch, sequences_lengths, logits, loss 

        loss_list = np.array(loss_list)
        loss_mean_list.append(np.mean(loss_list))
        print("Training loss mean =", np.mean(loss_list))
        print("Training loss std =", np.std(loss_list))

        full_val_loss = validation(valset_loader)
        val_list.append(full_val_loss)
        print("Validation loss =", full_val_loss)
        
        if full_val_loss < best_val_loss:
            
            torch.save(rnn.state_dict(), 'music_rnn_'+str(epoch_number)+'.pt')
            best_val_loss = full_val_loss


        plt.plot(list(range(epoch_number+1)), loss_mean_list)
        plt.plot(list(range(epoch_number+1)), val_list)
        plt.savefig('training.png')