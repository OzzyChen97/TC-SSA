# Evaluation Results

## 1. Summary of New Training Results

Based on the latest evaluation runs in `/workspace/ETC/eval_results`, here are the performance metrics for the four categories:

| Model | Dataset | Accuracy | AUC | Precision | Recall | F1 Score |
|-------|---------|----------|-----|-----------|--------|----------|
| **ResNet-50** | TCGA-BRCA | 0.8021 | 0.6958 | 0.6433 | 0.8021 | 0.7140 |
| **UNI** | TCGA-BRCA | 0.9271 | 0.9672 | 0.9259 | 0.9271 | 0.9263 |
| **ResNet-50** | TCGA-NSCLC | 0.7423 | 0.8315 | 0.7571 | 0.7423 | 0.7398 |
| **UNI** | TCGA-NSCLC | 0.8969 | 0.9881 | 0.9076 | 0.8969 | 0.8959 |

**Key Observations:**
- The **UNI encoder** significantly outperforms ResNet-50 on both datasets, achieving near 99% AUC on NSCLC and 97% on BRCA.
- **ResNet-50 on BRCA** shows signs of mode collapse (predicting only the majority class), resulting in an accuracy that matches the majority class prevalence (~80%) but a poor Recall for the minority class (0.0).

---

## 2. Training Visualization

Selected training plots showing the progression of Accuracy, AUC, and Loss.

### TCGA-BRCA (ResNet-50 vs UNI)

| ResNet-50 AUC | UNI AUC |
|:---:|:---:|
| ![BRCA R50 AUC](./amy_plots/brca-r50/auc_epoch.png) | ![BRCA UNI AUC](./amy_plots/brca-uni/auc_epoch.png) |

| ResNet-50 Loss | UNI Loss |
|:---:|:---:|
| ![BRCA R50 Loss](./amy_plots/brca-r50/loss_epoch.png) | ![BRCA UNI Loss](./amy_plots/brca-uni/loss_epoch.png) |


### TCGA-NSCLC (ResNet-50 vs UNI)

| ResNet-50 AUC | UNI AUC |
|:---:|:---:|
| ![NSCLC R50 AUC](./amy_plots/nsclc-r50/auc_epoch.png) | ![NSCLC UNI AUC](./amy_plots/nsclc-uni/auc_epoch.png) |

---

## 3. Analysis: BRCA-ResNet-50 AUC Decline

A detailed inspection of the training process for `brca-r50` reveals a problematic trend where the AUC may degrade or stagnate despite reasonable accuracy.

**Root Cause Analysis:**
The evaluation metrics show a confusion matrix of:
```
[[77, 0],
 [19, 0]]
```
This indicates **Mode Collapse** where the model predicts the negative class (0) for all samples. 
- **High Accuracy (80.2%)** is misleading; it simply reflects the class imbalance (77/96 samples are class 0).
- **Declining/Low AUC**: Since the model is not learning to discriminate and pushes all probabilities towards one class or becomes unconfident/random regarding the minority class, the ranking capability (AUC) suffers.
- **Possible Reasons**:
    1. **Severe Class Imbalance**: The dataset has significantly more negatives than positives. Without oversampling or weighted loss, the model falls into the local minimum of "always predict majority".
    2. **Encoder Capacity**: ResNet-50 might be struggling to extract robust features for this specific task compared to UNI, leading to faster overfitting to the majority class priors.
    3. **Hyperparameters**: The learning rate might be too high, preventing the model from settling into a solution that distinguishes the minority class.

**Recommendation**: Implement **Weighted Cross Entropy Loss** or perform **Oversampling** for the minority class to force the ResNet-50 model to learn features for Class 1.
