import torch

class VisualTrainer:
    def __init__(self, model, optimizer, criterion, device, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        for i, (x, y) in enumerate(loader):
            # x: [1, N, 768], y: [1, 1]
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            pred, _ = self.model(x)
            
            # Loss normalized by accumulation steps
            loss = self.criterion(pred, y) / self.accumulation_steps
            loss.backward()
            
            # Optimizer step only after N iterations
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps # Get back to original loss value
            
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                pred, _ = self.model(x)
                loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                
        val_loss = total_loss / len(loader)
                
        return val_loss