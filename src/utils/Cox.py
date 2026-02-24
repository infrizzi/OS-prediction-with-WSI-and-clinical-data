import torch
import torch.nn as nn

class CoxPHLoss(nn.Module):
    """
    Cox Partial Likelihood Loss.
    Modelliamo il rischio relativo: un output più alto indica un rischio maggiore
    (quindi una sopravvivenza più breve).
    """
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_pred, labels):
        """
        risk_pred: [B, 1] - Log-risk predetto
        labels: [B, 1] - Mesi di sopravvivenza (target)
        """
        # 1. Ordiniamo il batch per tempo di sopravvivenza crescente
        # I pazienti che muoiono prima sono all'inizio
        labels, indices = torch.sort(labels, dim=0)
        risk_pred = risk_pred[indices]

        # 2. Calcoliamo l'esponenziale del rischio (Hazard)
        exp_risk = torch.exp(risk_pred)

        # 3. Somma cumulativa inversa (set a rischio)
        # Per ogni paziente i, sommiamo i rischi di tutti i pazienti che vivono almeno quanto lui
        # Usiamo il flip per fare una somma cumulativa dal fondo (pazienti che vivono di più)
        sum_exp_risk = torch.flip(torch.cumsum(torch.flip(exp_risk, dims=[0]), dim=0), dims=[0])
        
        # 4. Log-Likelihood Parziale
        # loss = - media(risk - log(somma_rischi_a_rischio))
        loss = -torch.mean(risk_pred - torch.log(sum_exp_risk + 1e-8))
        return loss