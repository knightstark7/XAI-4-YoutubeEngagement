import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundPro(nn.Module):
    """
    Processes audio input: (batch, length_audio, 2).
    Converts it into a fixed-size representation.
    """
    def __init__(self):
        super(SoundPro, self).__init__()
        self.conv1d = nn.Conv1d(2, 64, kernel_size=5, stride=2, padding=2)  # Input channels = 2 (stereo), output = 64
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2d = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 256)  # Output a fixed representation

    def forward(self, x):
        # x shape: (batch, length_audio, 2)
        x = x.permute(0, 2, 1)  # Convert to (batch, 2, length_audio) for Conv1d
        x = F.relu(self.bn1(self.conv1d(x)))
        x = F.relu(self.bn2(self.conv2d(x)))
        x = x.mean(dim=2)  # Global average pooling over the temporal dimension
        x = self.fc(x)
        return x


class VideoPro(nn.Module):
    """
    Processes video input: (batch, length, M, N, 3).
    Converts it into a fixed-size representation.
    """
    def __init__(self):
        super(VideoPro, self).__init__()
        self.conv3d = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)  # 3D Conv for spatiotemporal processing
        self.bn1 = nn.BatchNorm3d(16)
        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.fc = nn.Linear(32 * 5 * 32 * 32, 256)  # Flattened and reduced spatial dimensions

    def forward(self, x):
        print(x.shape)
        
        x = F.relu(self.bn1(self.conv3d(x)))
        print(x.shape)
        
        x = F.relu(self.bn2(self.conv3d2(x)))
        print(x.shape)
        
        x = x.flatten(start_dim=1)  # Flatten spatial and temporal 
        print(x.shape)
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    """
    Combines processed video and audio representations and predicts two continuous outputs.
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(256 + 256, 128)  # Combine audio and video embeddings
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output two continuous values

    def forward(self, audio_rep, video_rep):
        # Combine audio and video representations
        x = torch.cat((audio_rep, video_rep), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Outputs in range [0, 1]
        return x


class Model(nn.Module):
    """
    Full model that integrates audio, video processing, and classification.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.audio_processor = SoundPro()
        self.video_processor = VideoPro()
        self.classifier = Classifier()

    def forward(self, video, audio):
        # Process video and audio separately
        video_rep = self.video_processor(video)  # Video representation
        audio_rep = self.audio_processor(audio)  # Audio representation

        # Combine and classify
        output = self.classifier(audio_rep, video_rep)
        return output


# Example Usage
if __name__ == "__main__":
    # Mock input shapes
    batch_size = 4
    video = torch.randn(batch_size, 3, 10, 64, 64)  # (batch, length, M, N, 3)
    audio = torch.randn(batch_size, 10000, 2)  # (batch, length_audio, 2)

    # Model initialization
    model = Model()
    output = model(video, audio)
    model.
    # Print output
    print("Output shape:", output.shape)  # Expected shape: (batch_size, 2)
    print("Output:", output)
