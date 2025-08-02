import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    """
    Simple linear probe (single linear layer without bias).
    """
    
    def __init__(self, input_size: int, num_classes: int, use_bias: bool = False):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, num_classes, bias=use_bias)

    def forward(self, x):
        return self.linear(x)


def sign(x: torch.Tensor) -> torch.Tensor:
    """Apply sign function to threshold outputs to -1 or 1."""
    return torch.where(x < 0, -1, 1)


def load_processed_data(save_dir: str) -> pd.DataFrame:
    """
    Load processed data from .npz files and create DataFrame.
    
    Args:
        save_dir: Directory containing processed .npz files
        
    Returns:
        DataFrame with loaded data
    """
    data = []
    
    for filename in os.listdir(save_dir):
        if filename.endswith('.npz'):
            # Load the file
            loaded_file = dict(np.load(os.path.join(save_dir, filename), allow_pickle=True))
            
            # Reshape the residual streams
            if not data:
                print(f"Residual shape: {loaded_file['residual_stream'].shape}")
            
            residual_unconditional = loaded_file['residual_stream'].reshape(24, 2, 1024)[:, 1, :]
            residual_conditional = loaded_file['residual_stream'].reshape(24, 2, 1024)[:, 0, :]
            
            # Extract genre
            genre = loaded_file['genre']
            
            # Create a dictionary for the current file's data
            file_data = {
                'filename': filename,
                'genre': genre,
            }
            
            # Add residuals to the dictionary
            for layer in range(24):
                file_data[f'residual_conditional_{layer + 1}'] = residual_conditional[layer, :]
                file_data[f'residual_unconditional_{layer + 1}'] = residual_unconditional[layer, :]
            
            # Append the file data to the list
            data.append(file_data)
    
    return pd.DataFrame(data)


def prepare_data_for_training(df: pd.DataFrame, layer: int, 
                            residual_type: str = 'conditional') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training linear probes.
    
    Args:
        df: DataFrame with processed data
        layer: Layer number to train probe on
        residual_type: Type of residual ('conditional' or 'unconditional')
        
    Returns:
        Tuple of (features, labels)
    """
    col = f"residual_{residual_type}_{layer}"
    X = normalize(np.stack(df[col].values), norm='l2')
    y = df["label"].values.astype(np.float32)
    
    return X, y


def train_probe_mse(X: np.ndarray, y: np.ndarray, 
                   input_size: int = 1024, num_classes: int = 1,
                   batch_size: int = 1024, num_epochs: int = 250,
                   learning_rate: float = 0.005, device: str = "cuda") -> Tuple[SimpleNN, List[float], List[float]]:
    """
    Train linear probe using MSE loss (without sigmoid/softmax).
    
    Args:
        X: Input features
        y: Target labels
        input_size: Size of input features
        num_classes: Number of output classes
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Tuple of (trained model, training losses, validation losses)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = SimpleNN(input_size=input_size, num_classes=num_classes, use_bias=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                # Calculate accuracy using sign function
                predicted = sign(outputs)
                total += y_batch.size(0)
                correct += torch.eq(predicted, y_batch).sum().item()
        
        val_loss /= total
        val_losses.append(val_loss)
        accuracy = correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.8f}, "
                  f"Val Loss: {val_loss:.8f}, Accuracy: {accuracy:.8f}")
    
    return model, train_losses, val_losses


def train_probe_cross_entropy(X: np.ndarray, y: np.ndarray, 
                             input_size: int = 1024, num_classes: int = 2,
                             batch_size: int = 1024, num_epochs: int = 250,
                             learning_rate: float = 0.005, device: str = "cuda") -> Tuple[SimpleNN, List[float], List[float]]:
    """
    Train linear probe using Cross Entropy loss.
    
    Args:
        X: Input features
        y: Target labels (should be integer class labels)
        input_size: Size of input features
        num_classes: Number of output classes
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Tuple of (trained model, training losses, validation losses)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Convert labels to integer class labels for cross entropy
    y_int = y.astype(np.int64)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.15, random_state=42, shuffle=True
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = SimpleNN(input_size=input_size, num_classes=num_classes, use_bias=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_loss /= total
        val_losses.append(val_loss)
        accuracy = correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.8f}, "
                  f"Val Loss: {val_loss:.8f}, Accuracy: {accuracy:.8f}")
    
    return model, train_losses, val_losses


def train_probes_all_layers(df: pd.DataFrame, residual_type: str = 'conditional',
                           loss_type: str = 'mse', num_epochs: int = 250) -> Dict[int, np.ndarray]:
    """
    Train linear probes for all layers.
    
    Args:
        df: DataFrame with processed data
        residual_type: Type of residual ('conditional' or 'unconditional')
        loss_type: Type of loss ('mse' or 'cross_entropy')
        num_epochs: Number of training epochs
        
    Returns:
        Dictionary mapping layer numbers to trained weights
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = list(range(1, 25))  # Layers 1 to 24
    weights_dict = {}
    
    print(f"Training {loss_type.upper()} probes for all layers...")
    
    for layer in layers:
        print(f"\nTraining probe for layer {layer}...")
        
        # Prepare data
        X, y = prepare_data_for_training(df, layer, residual_type)
        
        # Train probe based on loss type
        if loss_type == 'mse':
            model, train_losses, val_losses = train_probe_mse(
                X, y, num_epochs=num_epochs, device=device
            )
        elif loss_type == 'cross_entropy':
            # For cross entropy, we need to convert labels to class indices
            # Assuming binary classification: -1 -> 0, 1 -> 1
            y_class = ((y + 1) / 2).astype(np.int64)
            model, train_losses, val_losses = train_probe_cross_entropy(
                X, y_class, num_epochs=num_epochs, device=device
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Store weights
        weights_dict[layer] = model.linear.weight.detach().cpu().numpy()
        
        print(f"Layer {layer} training completed. Final accuracy: {val_losses[-1]:.8f}")
    
    return weights_dict


def evaluate_probe_performance(df: pd.DataFrame, weights_dict: Dict[int, np.ndarray],
                             residual_type: str = 'conditional') -> Dict[str, List[float]]:
    """
    Evaluate probe performance across all layers.
    
    Args:
        df: DataFrame with processed data
        weights_dict: Dictionary of trained weights
        residual_type: Type of residual to evaluate
        
    Returns:
        Dictionary with accuracy and loss lists for each layer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = list(range(1, 25))
    
    accuracies = []
    losses = []
    
    for layer in layers:
        # Prepare data
        X, y = prepare_data_for_training(df, layer, residual_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, shuffle=True
        )
        
        # Convert to tensors
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)
        
        # Create model with trained weights
        model = SimpleNN(input_size=1024, num_classes=1, use_bias=False).to(device)
        model.linear.weight.data = torch.tensor(weights_dict[layer], dtype=torch.float32).to(device)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            loss = nn.MSELoss()(outputs, y_test)
            
            # Calculate accuracy using sign function
            predicted = sign(outputs)
            accuracy = torch.eq(predicted, y_test).float().mean().item()
        
        accuracies.append(accuracy)
        losses.append(loss.item())
        
        print(f"Layer {layer}: Accuracy = {accuracy:.4f}, Loss = {loss.item():.4f}")
    
    return {'accuracies': accuracies, 'losses': losses}


def plot_results(layers: List[int], accuracies: List[float], losses: List[float],
                save_path: str = None):
    """
    Plot training results.
    
    Args:
        layers: List of layer numbers
        accuracies: List of accuracies
        losses: List of losses
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracies
    ax1.plot(layers, accuracies, marker='o')
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy by Layer')
    ax1.grid(True)
    ax1.set_xticks(layers)
    
    # Plot losses
    ax2.plot(layers, losses, marker='o', color='red')
    ax2.set_xlabel('Layer Number')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss by Layer')
    ax2.grid(True)
    ax2.set_xticks(layers)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def save_weights(weights_dict: Dict[int, np.ndarray], save_path: str):
    """
    Save trained weights to file.
    
    Args:
        weights_dict: Dictionary of trained weights
        save_path: Path to save weights
    """
    np.save(save_path, weights_dict, allow_pickle=True)
    print(f"Weights saved to: {save_path}")


def load_weights(load_path: str) -> Dict[int, np.ndarray]:
    """
    Load trained weights from file.
    
    Args:
        load_path: Path to load weights from
        
    Returns:
        Dictionary of loaded weights
    """
    weights_dict = np.load(load_path, allow_pickle=True).item()
    print(f"Weights loaded from: {load_path}")
    return weights_dict


if __name__ == "__main__":
    # Example usage
    print("Linear probes module loaded successfully!")
    print("Use train_probes_all_layers() to train probes for all layers.")
    print("Use evaluate_probe_performance() to evaluate trained probes.")
    print("Use plot_results() to visualize results.") 