import os
import pretty_midi
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# Paths for training and testing data
training_folder = "C:/Users/joyal/Desktop/Learning/Generative Ai/Music-Generation-with-Recurrent-Neural-Networks/Audio/Training"
testing_folder = "C:/Users/joyal/Desktop/Learning/Generative Ai/Music-Generation-with-Recurrent-Neural-Networks/Audio/Testing"

# Function to extract notes from a MIDI file
def extract_notes(midi_file):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(note.pitch)
        return notes
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return []

# Function to preprocess a folder of MIDI files
def preprocess_folder(folder_path):
    all_notes = []
    for file in os.listdir(folder_path):
        if file.endswith((".mid", ".midi")):
            notes = extract_notes(os.path.join(folder_path, file))
            all_notes.extend(notes)
    return all_notes

# Preprocess training and testing folders
print("Processing training data...")
training_notes = preprocess_folder(training_folder)
print("Processing testing data...")
testing_notes = preprocess_folder(testing_folder)

# Encode notes as integers
encoder = LabelEncoder()
all_notes_combined = training_notes + testing_notes
encoder.fit(all_notes_combined)

# Save the encoder
np.save("note_encoder.npy", encoder.classes_)

# Encode training and testing notes
training_encoded = encoder.transform(training_notes)
testing_encoded = encoder.transform(testing_notes)

# Input sequence length
sequence_length = 50

def create_sequences(encoded_notes):
    input_sequences = []
    output_labels = []
    for i in range(len(encoded_notes) - sequence_length):
        input_sequences.append(encoded_notes[i:i + sequence_length])
        output_labels.append(encoded_notes[i + sequence_length])
    return np.array(input_sequences), np.array(output_labels)

# Create sequences for training and testing
training_inputs, training_labels = create_sequences(training_encoded)
testing_inputs, testing_labels = create_sequences(testing_encoded)

# Save processed data
np.save("training_inputs.npy", training_inputs)
np.save("training_labels.npy", training_labels)
np.save("testing_inputs.npy", testing_inputs)
np.save("testing_labels.npy", testing_labels)

print(f"Training data: {len(training_inputs)} sequences")
print(f"Testing data: {len(testing_inputs)} sequences")

# Hyperparameters
input_size = 1  # Each note is represented by a single integer
hidden_size = 512  # Number of units in LSTM hidden layers
output_size = len(encoder.classes_)  # Total number of unique notes
num_epochs = 50
batch_size = 64
learning_rate = 0.001

# LSTM Model Definition
class MusicGenerationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicGenerationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output for prediction
        return out

# Instantiate the model
model = MusicGenerationLSTM(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i in range(0, len(training_inputs) - batch_size, batch_size):
        # Prepare input and target sequences
        input_seq = training_inputs[i:i + batch_size]
        target_seq = training_labels[i:i + batch_size]

        # Convert to tensors
        input_tensor = torch.Tensor(input_seq).view(-1, sequence_length, 1)  # Reshape for LSTM
        target_tensor = torch.LongTensor(target_seq)

        # Forward pass
        outputs = model(input_tensor)

        # Compute loss
        loss = criterion(outputs, target_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model and encoder
torch.save({
    'model_state_dict': model.state_dict(),
    'encoder_classes': encoder.classes_,
    'hidden_size': hidden_size,
    'sequence_length': sequence_length
}, 'music_generation_lstm.pth')

# Music Generation Function
def generate_music(model, start_sequence, num_generate=500):
    model.eval()  # Switch to evaluation mode
    generated_notes = start_sequence.copy()

    for _ in range(num_generate):
        input_seq = generated_notes[-sequence_length:]  # Get the last `sequence_length` notes
        input_tensor = torch.Tensor(input_seq).view(1, sequence_length, 1)  # Reshape for LSTM

        with torch.no_grad():
            predicted_note = model(input_tensor)

        # Get the predicted note (argmax)
        predicted_note = torch.argmax(predicted_note, dim=-1).item()

        # Append the predicted note to the generated sequence
        generated_notes.append(predicted_note)

    return generated_notes

# Example: Generate music
start_sequence = training_encoded[:sequence_length]  # Use the first `sequence_length` notes
generated_music = generate_music(model, start_sequence, num_generate=500)

# Convert generated music back to notes
decoded_notes = encoder.inverse_transform(generated_music)
print(decoded_notes)
