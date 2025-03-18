import os, torch, torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.tensorboard.writer import SummaryWriter
from typing import cast
from torch.multiprocessing import freeze_support

class SPEECHCOMMANDSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Subset):
        self.dataset = dataset
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=64
        )
        self.amplitude_to_db = AmplitudeToDB()
        self.label_mapping = self.generate_label_mapping()
        print("Generated label mapping:", self.label_mapping)

    def generate_label_mapping(self):
        unique_labels = set()
        for _, _, label, *_ in self.dataset:  # Iterate through the dataset
            unique_labels.add(label)
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int | list[int]) -> tuple[torch.Tensor, int]:
        try:
            waveform, sample_rate, label, *_ = self.dataset[idx]
            if label in self.label_mapping:
                label = self.label_mapping[label]
            else:
                raise ValueError(f"Unexpected label: {label}")
            waveform = self.pad_waveform(waveform)
            spectrogram = self.amplitude_to_db(self.mel_spectrogram(waveform))
            current_size = spectrogram.size(2)
            target_size = 128
            if current_size < target_size:
                padding_size = target_size - current_size
                spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_size))
            elif current_size > target_size:
                spectrogram = spectrogram[:, :, :target_size]
            return spectrogram, label
        except Exception as e:
            print(f"Error processing sample at index {idx}: {e}")
            raise

    @staticmethod
    def pad_waveform(waveform: torch.Tensor, length: int = 16000) -> torch.Tensor:
        if waveform.size(-1) < length:
            waveform = torch.nn.functional.pad(waveform, (0, length - waveform.size(-1)))
        else:
            waveform = waveform[:, :length]
        return waveform


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CRNN, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.gru1 = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        self.gru2 = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adaptive_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 1, 3)
        batch_size, seq_len, channels, features = x.shape
        x = x.reshape(batch_size, seq_len, -1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    freeze_support()
    data_path = os.path.normpath("data")
    os.makedirs(data_path, exist_ok=True)
    print("Available torchaudio backends:", torchaudio.list_audio_backends())
    try:
        # Try to set the backend
        if "sox_io" in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend("sox_io")
        elif "soundfile" in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend("soundfile")
        print("Selected torchaudio backend:", torchaudio.get_audio_backend())

        # Load the data
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root=data_path, download=True)
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        total_size = len(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        raise
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset = SPEECHCOMMANDSDataset(train_dataset)
    val_dataset = SPEECHCOMMANDSDataset(val_dataset)
    test_dataset = SPEECHCOMMANDSDataset(test_dataset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


    num_classes = len(train_dataset.label_mapping)
    model = CRNN(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

    writer = SummaryWriter()
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss: float = 0
        total_correct: int = 0
        total_samples: int = 0

        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_accuracy = total_correct / total_samples
        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

        model.eval()
        val_loss: float = 0
        val_correct: int = 0
        val_samples: int = 0

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples += labels.size(0)

        val_accuracy = val_correct / val_samples
        writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        print(f"Validation - Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")

    model.eval()
    total_correct: int = 0
    total_samples: int = 0

    print("\n" + "=" * 50)
    print("EVALUATION ON TEST DATASET")
    print("=" * 50)

    try:
        with torch.no_grad():
            for i, (spectrograms, labels) in enumerate(test_loader):
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                outputs = model(spectrograms)
                _, predicted = torch.max(outputs, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)

                total_correct += batch_correct
                total_samples += batch_total
                if i % 5 == 0:
                    print(f"Batch {i}/{len(test_loader)}: Accuracy so far: {total_correct / total_samples:.4f}")

        test_accuracy = total_correct / total_samples
        print("\nFINAL RESULTS:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Total correct: {total_correct} out of {total_samples}")
        print("=" * 50)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()

    torch.save(model.state_dict(), "model.pth")
    print("Model saved.")
