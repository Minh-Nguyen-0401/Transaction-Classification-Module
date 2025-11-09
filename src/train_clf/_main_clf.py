import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class TriTowerClassifier(nn.Module):
    """Multi-layer classifier combining context, sender, and recipient embeddings"""
    
    def __init__(self, d_in, n_classes, hidden_dims=[512, 256], dropout_rates=[0.3, 0.2]):
        super().__init__()
        
        layers = []
        prev_dim = d_in
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ClassifierTrainer:
    """Trainer for tri-tower classifier"""
    
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.best_state = None
    
    def train_epoch(self, dataloader, desc="Training"):
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for xb, yb in pbar:
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return epoch_loss / n_batches
    
    def valid_epoch(self, dataloader, desc="Validation"):
        self.model.eval()
        epoch_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(dataloader, desc=desc, leave=False)
        with torch.no_grad():
            for xb, yb in pbar:
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                epoch_loss += loss.item()
                n_batches += 1
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # ROC AUC (one-vs-rest for multi-class)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = 0.0
        
        metrics = {
            'loss': epoch_loss / n_batches,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'roc_auc_ovr': roc_auc
        }
        
        return metrics
    
    def train(self, train_dl, val_dl=None, epochs=8, patience=3, val_every=1, verbose=True):
        patience_counter = 0
        
        pbar_epochs = tqdm(range(epochs), desc="Training Progress") if verbose else range(epochs)
        
        for epoch in pbar_epochs:
            train_loss = self.train_epoch(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
            self.history['train'].append(train_loss)
            
            msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}"
            
            if val_dl is not None and (epoch + 1) % val_every == 0:
                val_metrics = self.valid_epoch(val_dl, desc=f"Validation {epoch+1}/{epochs}")
                self.history['val'].append(val_metrics)
                
                val_loss = val_metrics['loss']
                msg += f" | Val Loss: {val_loss:.6f}"
                msg += f" | Acc: {val_metrics['accuracy']:.4f}"
                msg += f" | F1: {val_metrics['f1_macro']:.4f}"
                msg += f" | ROC-AUC: {val_metrics['roc_auc_ovr']:.4f}"
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    msg += " [BEST]"
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            if isinstance(pbar_epochs, tqdm):
                                pbar_epochs.write(msg)
                                pbar_epochs.write(f"Early stopping at epoch {epoch+1}")
                            else:
                                print(msg)
                                print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if verbose:
                if isinstance(pbar_epochs, tqdm):
                    pbar_epochs.write(msg)
                else:
                    print(msg)
        
        if val_dl is not None and self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            if verbose:
                print(f"Restored best model with val loss: {self.best_val_loss:.6f}")
        
        return self.history
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
