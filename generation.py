import pretty_midi
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from training import RNN


def write_to_midi(sample, filename):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=True)

    clock = 0
    cur_note_start = 0
    # Iterate over note names, which will be converted to note number later
    for step in sample:
        clock = clock + 1.0 / 4
        step = step.nonzero()[0].tolist()
        for ind in step:
            note = pretty_midi.Note(velocity=50, pitch=ind, start=cur_note_start, end=clock)
            instrument.notes.append(note)
        cur_note_start = clock

    midi.instruments.append(instrument)
    midi.write(filename)


def sample_from_piano_rnn(rnn, sample_length=4, temperature=1, starting_sequence=None):

    if starting_sequence is None:
        current_sequence_input = torch.zeros(1, 1, 128)
        current_sequence_input[0, 0, 36] = 1
        current_sequence_input = Variable(current_sequence_input.cuda())

    final_output_sequence = [current_sequence_input.data.squeeze(1)]
    hidden = None

    for i in range(sample_length):
        output, hidden = rnn(current_sequence_input, [1], hidden)
        probabilities = torch.nn.functional.softmax(output.div(temperature), dim=1)
        current_sequence_input = torch.multinomial(probabilities.data, 1).squeeze().unsqueeze(0).unsqueeze(1)
        current_sequence_input = Variable(current_sequence_input.float())
        final_output_sequence.append(current_sequence_input.data.squeeze(1))

    sampled_sequence = torch.cat(final_output_sequence, dim=0).cpu().numpy()
    
    return sampled_sequence


temperature = 5

rnn = RNN(input_size=128, hidden_size=512, num_classes=128).cuda()
rnn.load_state_dict(torch.load('music_rnn.pt'))

sample = sample_from_piano_rnn(rnn, sample_length=200, temperature=temperature).transpose()
plt.imshow(sample)
plt.savefig('sample_'+str(temperature)+'.png')

write_to_midi(sample.transpose(), 'sample_'+str(temperature)+'.midi')
