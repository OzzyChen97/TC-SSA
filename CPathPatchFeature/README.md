---
language:
- en
license: apache-2.0
size_categories:
- 100B<n<1T
tags:
- medical
- pathology
task_categories:
- image-feature-extraction
---

# CPathPatchFeature: Pre-extracted WSI Features for Computational Pathology

Paper: [Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](https://huggingface.co/papers/2506.02408)
Code: [https://github.com/DearCaat/E2E-WSI-ABMILX](https://github.com/DearCaat/E2E-WSI-ABMILX)

## Dataset Summary

This dataset provides a comprehensive collection of pre-extracted features from Whole Slide Images (WSIs) for various cancer types, designed to facilitate research in computational pathology. The features are extracted using multiple state-of-the-art encoders, offering a rich resource for developing and evaluating Multiple Instance Learning (MIL) models and other deep learning architectures.


The repository contains features for the following public datasets:
- **PANDA**: Prostate cANcer graDe Assessment
- **TCGA-BRCA**: Breast Cancer in TCGA
- **TCGA-NSCLC**: Non-Small Cell Lung Cancer in TCGA
- **TCGA-BLCA**: Bladder Cancer in TCGA
- **CAMELYON**: Cancer Metastases in Lymph Nodes
- **CPTAC-NSCLC**: Non-Small Cell Lung Cancer in CPTAC

## Dataset Structure

The features for each WSI dataset are organized into subdirectories. Each subdirectory contains the features extracted by a specific encoder, along with the corresponding patch coordinates.

### Feature Encoders
The following encoders were used to generate the features:
- **UNI**: A vision-language pretrained model for pathology ([UNI by Chen et al.](https://www.nature.com/articles/s41591-024-02857-3)).
- **CHIEF**: A feature extractor based on self-supervised learning for pathology ([CHIEF by Wang et al.](https://www.nature.com/articles/s41586-024-07894-z)).
- **GIGAP**: A Giga-Pixel vision model for pathology ([GigaPath by Xu et al.](https://www.nature.com/articles/s41586-024-07441-w)).
- **R50**: A ResNet-50 model pre-trained on ImageNet.

Some data may not be fully organized yet. If you have specific needs or questions, please feel free to open an issue in the community tab.

## How to Use

You can load and access the dataset using the Hugging Face `datasets` library or by cloning the repository with Git LFS.

### Using the `datasets` Library

To load the data, you can use the following Python code:

```python
from datasets import load_dataset

# Load a specific subset (e.g., PANDA)
# Note: You may need to specify the data files manually depending on the configuration.
# Example for a hypothetical configuration named 'panda'
# ds = load_dataset("your-username/CPathPatchFeature", name="panda")

# For datasets with this structure, it's often easier to download and access files directly.
# We recommend using Git LFS for a complete download.
````

*Note: Due to the heterogeneous structure (mixed zipped and unzipped files), direct loading with `load_dataset` might be complex. The recommended approach is to clone the repository.*

### Using Git LFS

First, ensure you have Git LFS installed and configured:

```bash
git lfs install
```

Then, clone the dataset repository:

```bash
git clone https://huggingface.co/datasets/Dearcat/CPathPatchFeature
```

### Citation
This dataset has been used in the following publications. If you find it useful for your research, please consider citing them:

```bibtex
@misc{tang2025revisitingdatachallengescomputational,
      title={Revisiting Data Challenges of Computational Pathology: A Pack-based Multiple Instance Learning Framework}, 
      author={Wenhao Tang and Heng Fang and Ge Wu and Xiang Li and Ming-Ming Cheng},
      year={2025},
      eprint={2509.20923},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2509.20923](https://arxiv.org/abs/2509.20923)}, 
}

@misc{tang2025multipleinstancelearningframework,
      title={Multiple Instance Learning Framework with Masked Hard Instance Mining for Gigapixel Histopathology Image Analysis}, 
      author={Wenhao Tang and Sheng Huang and Heng Fang and Fengtao Zhou and Bo Liu and Qingshan Liu},
      year={2025},
      eprint={2509.11526},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2509.11526](https://arxiv.org/abs/2509.11526)}, 
}

@misc{tang2025revisitingendtoendlearningslidelevel,
      title={Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology}, 
      author={Wenhao Tang and Rong Qin and Heng Fang and Fengtao Zhou and Hao Chen and Xiang Li and Ming-Ming Cheng},
      year={2025},
      eprint={2506.02408},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2506.02408](https://arxiv.org/abs/2506.02408)}, 
}
```