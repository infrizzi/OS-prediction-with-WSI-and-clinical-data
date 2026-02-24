import torch


class MCATTrainer:
    def __init__(self, model, optimizer, criterion, device, accumulation_steps=1):
        """
        model forward returns: pred, info_dict
        pred: [B, 1]
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps

    def _to_device_and_shape(self, clin_x, vis_x, y):
        clin_x = clin_x.to(self.device)
        vis_x = vis_x.to(self.device)
        y = y.to(self.device)

        # Ensure shapes:
        # clin_x -> [B, D_in]
        if clin_x.dim() == 1:
            clin_x = clin_x.unsqueeze(0)

        # vis_x -> [B, N, D_vis]
        if vis_x.dim() == 2:
            vis_x = vis_x.unsqueeze(0)

        # y -> [B, 1]
        if y.dim() == 0:
            y = y.view(1, 1)
        elif y.dim() == 1:
            y = y.unsqueeze(1)

        return clin_x, vis_x, y

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        # Buffer per accumulare predizioni e label durante l'accumulation step
        preds_buffer = []
        ys_buffer = []

        for i, batch in enumerate(loader):
            clin_x, vis_x, y = batch
            clin_x, vis_x, y = self._to_device_and_shape(clin_x, vis_x, y)

            # Eseguiamo il forward
            pred, _ = self.model(clin_x, vis_x)
            
            # Accumuliamo nei buffer invece di calcolare la loss subito
            preds_buffer.append(pred)
            ys_buffer.append(y)

            # Quando raggiungiamo gli accumulation steps, calcoliamo la loss sul "mini-batch" accumulato
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                # Concateniamo tutto il buffer: es. [256, 1]
                batch_preds = torch.cat(preds_buffer, dim=0)
                batch_ys = torch.cat(ys_buffer, dim=0)

                # Calcoliamo la Cox Loss sull'intero set accumulato
                loss = self.criterion(batch_preds, batch_ys)
                
                # Backward unico per l'intero set
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item() * len(preds_buffer)
                
                # Reset buffer
                preds_buffer = []
                ys_buffer = []

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        
        # Anche in validazione, dobbiamo calcolare la loss su tutto il set o su batch consistenti
        # Raccogliamo tutto per avere una stima corretta della Cox Loss
        all_preds = []
        all_ys = []

        for batch in loader:
            clin_x, vis_x, y = batch
            clin_x, vis_x, y = self._to_device_and_shape(clin_x, vis_x, y)
            pred, _ = self.model(clin_x, vis_x)
            all_preds.append(pred)
            all_ys.append(y)
        
        full_preds = torch.cat(all_preds, dim=0)
        full_ys = torch.cat(all_ys, dim=0)
        
        loss = self.criterion(full_preds, full_ys)
        return loss.item()