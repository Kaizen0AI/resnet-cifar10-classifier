### Experiment 1: Baseline ResNet

Result:
- Train: 95.85%
- Val: 78.95%
- Loss: 0.7336

Observation:
Large train-validation gap.

### Experiment 2: Baseline + Data Augmentation

Result:
- Train: 95.85% → 90.93%
- Val: 78.95% → 85.15%
- Loss: 0.7336 → 0.4722

Conclusion:
Data augmentation provided very significant imprvement in generalization, but the variance is still present.

### Experiment 3: Add Dropout (0.1)

Result:
- Train: 90.93% → 91.35%
- Val: 85.15% → 85.57%
- Loss: 0.4722 → 0.4695

Conclusion:
Dropout provided a small improvement in generalization, but the effect was modest compared to data augmentation. The architecture itself is likely the next limiting factor.

### Experiment 4: Remove MaxPool layer

Observation:
The first residual stage operates on 32×32 feature maps instead of 16×16.

Training time increased from ~2 minutes/epoch to over 6 minutes/epoch.

Experiment discontinued due to computational cost.

### Experiment 5: Width of last Residual Block Reduced (192 → 160)

Result:
- Train: 91.35% → 90.55%
- Val: 85.57% → 84.72%
- Loss: 0.4695 → 0.4714

Observation:
Reduced the number of filters in last two residual bloacks from 192 tp 160, to check if it affects the overfitting problem.

Training time decreased to slightly less than 2 minutes(~102 secs)/epoch.

Generalization gap got slightly worse(5.77% → 5.83%) and accuracies also took a hit. 



| Experiment | Best Epoch | Train Acc | Val Acc | Test Acc |
| ---------- | ---------: | --------: | ------: | -------: |
| No Aug     |          4 |    95.85% |  78.95% |   78.56% |
| Data Aug   |         21 |    90.93% |  85.15% |   83.97% |
| Dropout    |         24 |    91.35% |  85.57% |   85.01% |
| Width160   |         22 |    90.55% |  84.72% |   84.47% |

