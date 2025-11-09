import torch
from torch import nn
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class CtxMLP(nn.Module):
    def __init__(self, d_in, d_hidden=256, d_out=128, p_drop=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out),
            nn.LayerNorm(d_out)
        )

    def forward(self, x):
        return self.enc(x)

class CtxAutoEncoder(nn.Module):
    def __init__(self, encoder: CtxMLP, d_hidden=256):
        super().__init__()
        self.encoder = encoder
        self.d_in = encoder.enc[0].in_features
        # Find the last Linear layer in the encoder to get output dimension
        for layer in reversed(encoder.enc):
            if isinstance(layer, nn.Linear):
                self.d_out = layer.out_features
                break
        self.decoder = nn.Sequential(
            nn.Linear(self.d_out, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, self.d_in)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class CtxTrainer:
    def __init__(self, autoencoder: CtxAutoEncoder, lr=1e-3):
        self.model = autoencoder
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.best_state = None
    
    def train_epoch(self, dataloader, desc="Training"):
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for xb, in pbar:
            xr = self.model(xb)
            loss = self.loss_fn(xr, xb)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / n_batches
        return avg_loss
    
    def valid_epoch(self, dataloader, desc="Validation"):
        self.model.eval()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=desc, leave=False)
        with torch.no_grad():
            for xb, in pbar:
                xr = self.model(xb)
                loss = self.loss_fn(xr, xb)
                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / n_batches
        return avg_loss
    
    def train(self, train_dl, val_dl=None, epochs=5, patience=3, val_every=1, verbose=True):
        patience_counter = 0
        
        pbar_epochs = tqdm(range(epochs), desc="Training Progress") if verbose else range(epochs)
        
        for epoch in pbar_epochs:
            train_loss = self.train_epoch(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
            self.history['train'].append(train_loss)
            
            msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}"
            
            if val_dl is not None and (epoch + 1) % val_every == 0:
                val_loss = self.valid_epoch(val_dl, desc=f"Validation {epoch+1}/{epochs}")
                self.history['val'].append(val_loss)
                msg += f" - Val Loss: {val_loss:.6f}"
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    msg += " --- NEW BEST MODEL ---"
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
                    logger.info(msg)
        
        if val_dl is not None and self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            if verbose:
                logger.info(f"Restored best model with val loss: {self.best_val_loss:.6f}")
        
        return self.history
    
    def save_autoencoder(self, path):
        """Save the full autoencoder model"""
        torch.save(self.model.state_dict(), path)
    
    def save_encoder(self, path):
        """Save only the encoder part"""
        torch.save(self.model.encoder.state_dict(), path)
    
    def load_autoencoder(self, path):
        """Load the full autoencoder model"""
        self.model.load_state_dict(torch.load(path))
    
    def load_encoder(self, path):
        """Load only the encoder part"""
        self.model.encoder.load_state_dict(torch.load(path))

class SndEncoder(nn.Module):
    def __init__(self, tt_vocab, ch_vocab, num_dim=3, tt_emb=16, ch_emb=8, hidden=128, p_drop=0.1):
        super().__init__()
        self.tt = nn.Embedding(tt_vocab, tt_emb)
        self.ch = nn.Embedding(ch_vocab, ch_emb)
        self.dropout = nn.Dropout(p_drop)
        self.gru = nn.GRU(num_dim + tt_emb + ch_emb, hidden, batch_first=True)
        self.post = nn.Sequential(nn.LayerNorm(hidden))
        
    def forward(self, x_num, x_tt, x_ch):
        x = torch.cat([x_num, self.tt(x_tt), self.ch(x_ch)], dim=-1)
        x = self.dropout(x)
        _, h = self.gru(x)
        return self.post(h[-1])


class SndTrainer:
    """Trainer for SndEncoder with multi-task learning (tranx_type + amount prediction)"""
    def __init__(self, encoder: SndEncoder, n_tranx_types, n_amount_bins=5, lr=1e-3):
        self.encoder = encoder
        # Multi-task heads
        self.head_tranx = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder.gru.hidden_size, n_tranx_types)
        )
        self.head_amount = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder.gru.hidden_size, n_amount_bins)
        )
        
        # Optimizer for all parameters
        all_params = list(encoder.parameters()) + \
                     list(self.head_tranx.parameters()) + \
                     list(self.head_amount.parameters())
        self.opt = torch.optim.Adam(all_params, lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.best_state = None
    
    def train_epoch(self, dataloader, desc="Training"):
        self.encoder.train()
        self.head_tranx.train()
        self.head_amount.train()
        
        epoch_loss = 0.0
        epoch_loss_t = 0.0
        epoch_loss_a = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for xn, xt, xc, yt, ya in pbar:
            # Forward
            h = self.encoder(xn, xt, xc)
            loss_t = self.loss_fn(self.head_tranx(h), yt)
            loss_a = self.loss_fn(self.head_amount(h), ya)
            loss = loss_t + loss_a
            
            # Backward
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            epoch_loss += loss.item()
            epoch_loss_t += loss_t.item()
            epoch_loss_a += loss_a.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lt': f'{loss_t.item():.4f}',
                'la': f'{loss_a.item():.4f}'
            })
        
        avg_loss = epoch_loss / n_batches
        avg_loss_t = epoch_loss_t / n_batches
        avg_loss_a = epoch_loss_a / n_batches
        return avg_loss, avg_loss_t, avg_loss_a
    
    def valid_epoch(self, dataloader, desc="Validation"):
        self.encoder.eval()
        self.head_tranx.eval()
        self.head_amount.eval()
        
        epoch_loss = 0.0
        epoch_loss_t = 0.0
        epoch_loss_a = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=desc, leave=False)
        with torch.no_grad():
            for xn, xt, xc, yt, ya in pbar:
                h = self.encoder(xn, xt, xc)
                loss_t = self.loss_fn(self.head_tranx(h), yt)
                loss_a = self.loss_fn(self.head_amount(h), ya)
                loss = loss_t + loss_a
                
                epoch_loss += loss.item()
                epoch_loss_t += loss_t.item()
                epoch_loss_a += loss_a.item()
                n_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lt': f'{loss_t.item():.4f}',
                    'la': f'{loss_a.item():.4f}'
                })
        
        avg_loss = epoch_loss / n_batches
        avg_loss_t = epoch_loss_t / n_batches
        avg_loss_a = epoch_loss_a / n_batches
        return avg_loss, avg_loss_t, avg_loss_a
    
    def train(self, train_dl, val_dl=None, epochs=5, patience=3, val_every=1, verbose=True):
        patience_counter = 0
        
        pbar_epochs = tqdm(range(epochs), desc="Training Progress") if verbose else range(epochs)
        
        for epoch in pbar_epochs:
            train_loss, train_loss_t, train_loss_a = self.train_epoch(
                train_dl, desc=f"Epoch {epoch+1}/{epochs}"
            )
            self.history['train'].append({
                'total': train_loss,
                'tranx': train_loss_t,
                'amount': train_loss_a
            })
            
            msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} (lt: {train_loss_t:.4f}, la: {train_loss_a:.4f})"
            
            if val_dl is not None and (epoch + 1) % val_every == 0:
                val_loss, val_loss_t, val_loss_a = self.valid_epoch(
                    val_dl, desc=f"Validation {epoch+1}/{epochs}"
                )
                self.history['val'].append({
                    'total': val_loss,
                    'tranx': val_loss_t,
                    'amount': val_loss_a
                })
                msg += f" - Val Loss: {val_loss:.6f} (lt: {val_loss_t:.4f}, la: {val_loss_a:.4f})"
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_state = {
                        'encoder': {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()},
                        'head_tranx': {k: v.cpu().clone() for k, v in self.head_tranx.state_dict().items()},
                        'head_amount': {k: v.cpu().clone() for k, v in self.head_amount.state_dict().items()}
                    }
                    patience_counter = 0
                    msg += " --- NEW BEST MODEL ---"
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            if isinstance(pbar_epochs, tqdm):
                                pbar_epochs.write(msg)
                                pbar_epochs.write(f"Early stopping at epoch {epoch+1}")
                            else:
                                logger.info(msg)
                                logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            if verbose:
                if isinstance(pbar_epochs, tqdm):
                    pbar_epochs.write(msg)
                else:
                    logger.info(msg)
        
        # Restore best model
        if val_dl is not None and self.best_state is not None:
            self.encoder.load_state_dict(self.best_state['encoder'])
            self.head_tranx.load_state_dict(self.best_state['head_tranx'])
            self.head_amount.load_state_dict(self.best_state['head_amount'])
            if verbose:
                logger.info(f"Restored best model with val loss: {self.best_val_loss:.6f}")
        
        return self.history
    
    def save_encoder(self, path):
        """Save only the encoder"""
        torch.save(self.encoder.state_dict(), path)
    
    def save_all(self, encoder_path, head_tranx_path=None, head_amount_path=None):
        """Save encoder and optionally the prediction heads"""
        torch.save(self.encoder.state_dict(), encoder_path)
        if head_tranx_path:
            torch.save(self.head_tranx.state_dict(), head_tranx_path)
        if head_amount_path:
            torch.save(self.head_amount.state_dict(), head_amount_path)
    
    def load_encoder(self, path):
        """Load encoder weights"""
        self.encoder.load_state_dict(torch.load(path))
