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

        for i, batch in enumerate(loader):
            clin_x, vis_x, y = batch
            clin_x, vis_x, y = self._to_device_and_shape(clin_x, vis_x, y)

            pred, _ = self.model(clin_x, vis_x)
            loss = self.criterion(pred, y) / self.accumulation_steps
            loss.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            clin_x, vis_x, y = batch
            clin_x, vis_x, y = self._to_device_and_shape(clin_x, vis_x, y)

            pred, _ = self.model(clin_x, vis_x)
            loss = self.criterion(pred, y)
            total_loss += loss.item()

        return total_loss / len(loader)