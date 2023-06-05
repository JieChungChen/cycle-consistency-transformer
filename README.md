##  Cycle Consistency Transformer for RUL Prediction

### Environment
```
torch==1.11.0
```

### Usage
* Feature Extraction
```
python preprocessing.py
```

### Cycle Consistency Loss
![image](https://github.com/JieChungChen/cycle-consistency-transformer/assets/120712134/abd9176f-398c-4676-b3a7-307a418971dd)

```
class Cycle_Consistency_Loss(nn.Module):
    def __init__(self, penalty=0.01):
        super(Cycle_Consistency_Loss, self).__init__()
        self.penalty = penalty

    def forward(self, seq, src_len, combinations):
        loss_i, loss_j = 0, 0
        src_len = src_len//4
        for c in combinations: # the combinations in a batch
            seq1, seq2 = seq[c[0], :src_len[c[0]]], seq[c[1], :src_len[c[1]]]
            for i, p in enumerate(seq1):
                d1 = torch.sum(torch.square(seq2-p.repeat(len(seq2), 1)), dim=1)
                alpha = F.softmin(d1, dim=0).reshape(-1, 1)
                snn = torch.sum(alpha.repeat(1, 32)*seq2, dim=0) #  soft nearest neighbor
                d2 = torch.sum(torch.square(seq1-snn.repeat(len(seq1), 1)), dim=1)
                beta = F.softmin(d2, dim=0)
                u_id = torch.dot(beta, torch.arange(len(seq1)).cuda().float())
                std_id = torch.dot(beta, torch.square(torch.arange(len(seq1)).cuda().float()-u_id))
                loss_i+=(torch.square(i-u_id)/std_id)+self.penalty*torch.log(torch.sqrt(std_id))
            for j, q in enumerate(seq2):
                d1 = torch.sum(torch.square(seq1-q.repeat(len(seq1), 1)), dim=1)
                alpha = F.softmin(d1, dim=0).reshape(-1, 1)
                snn = torch.sum(alpha.repeat(1, 32)*seq1, dim=0) #  soft nearest neighbor
                d2 = torch.sum(torch.square(seq2-snn.repeat(len(seq2), 1)), dim=1)
                beta = F.softmin(d2, dim=0)
                u_id = torch.dot(beta, torch.arange(len(seq2)).cuda().float())
                std_id = torch.dot(beta, torch.square(torch.arange(len(seq2)).cuda().float()-u_id))
                loss_j+=(torch.square(j-u_id)/std_id)+self.penalty*torch.log(torch.sqrt(std_id))
        return (loss_i+loss_j)/len(combinations)
```
* the code above equals to: <img src="https://github.com/JieChungChen/cycle-consistency-transformer/assets/120712134/b9e5f0b7-98be-410d-957a-2a7ca96e8b35" width="300" alt="img01"/>

