# Music Generation with an LSTM

A learning project that trains a recurrent neural network (LSTM) in PyTorch on classical piano MIDI files, then uses it to generate new note sequences.

## How it works

1. Reads every MIDI file in `Audio/Training` and `Audio/Testing` with `pretty_midi` and pulls out the note pitches.
2. Encodes notes as integers with scikit-learn's `LabelEncoder` and builds sequences of 50 notes, each paired with the next note as the label.
3. Trains an LSTM (512 hidden units, then a linear layer over all possible notes) with cross-entropy loss and Adam, for 50 epochs.
4. Saves the model to `music_generation_lstm.pth`, then generates new notes one at a time from a starting sequence.

## Files

| Path | Contents |
|---|---|
| `Music Generation with LSTM.py` | Preprocessing, training and generation |
| `Audio/Training`, `Audio/Testing` | MIDI files used for training and testing |
| `*.npy` | Preprocessed note sequences and the saved note encoder (some stored with Git LFS) |

## Tech

Python 3, PyTorch, pretty_midi, NumPy, scikit-learn

## How to run

```bash
pip install torch pretty_midi numpy scikit-learn
python "Music Generation with LSTM.py"
```

Before running, change `training_folder` and `testing_folder` at the top of the script to the `Audio/Training` and `Audio/Testing` folders of your copy. Some files are stored with Git LFS, so run `git lfs pull` after cloning.

## Data

The MIDI files appear to come from the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) (Google Magenta), which is licensed under CC BY-NC-SA 4.0.

## Results

No generated music or training figures are included in the repository. Running the script prints the loss for each epoch.
