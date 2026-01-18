## Introduction
We present SlideChat, the first open-source vision-language assistant capable of understanding gigapixel whole-slide images. To systematic0ally evaluate the performance of SlideChat, we developed SlideBench, a comprehensive benchmark comprising three components: SlideBench-Caption, SlideBench-VQA (TCGA), and SlideBench-VQA (BCNB).

1. SlideBench-Caption: This test set includes 734 WSIs from the TCGA dataset, providing a foundation to evaluate SlideChat's proficiency in generating accurate and coherent descriptions of WSIs.

2. SlideBench-VQA(TCGA): Designed for closed-set visual question answering, this subset evaluates multiple aspects of SlideChat’s performance with WSIs. After meticulous filtering by large language models (LLMs) and expert pathologists, SlideBench-VQA (TCGA) includes 7,827 VQA pairs across 13 categories.

3. SlideBench-VQA(BCNB): To further assess generalization capabilities, we incorporated the Early Breast Cancer Core-Needle Biopsy (BCNB) dataset, featuring a diverse patient population and a range of clinical task labels. By rephrasing classification objectives as questions and transforming multi-class labels into selectable options, we formatted the BCNB dataset as VQA pairs, creating a set of 7,247 VQAs under SlideBench-VQA (BCNB).

## About Data
The **SlideBench-VQA (BCNB).csv** file provides detailed testing information, including Slide(patient_id), Task, Question, Answer options (A, B, C, D), and the correct answer. Additionally, we provide extracted features for 1,058 WSIs in **BCNB_patch_feat.tar.gz** using the pre-trained CONCH(http://huggingface.co/MahmoodLab/conch) model. Each file contains 512-dimensional feature representations for patches within the WSI, along with corresponding spatial location information. The files are named by patient ID; for example, "1.csv" corresponds to patient ID 1. For more information on the original dataset and preprocessing steps, please refer to https://bupt-ai-cz.github.io/BCNB/.


## Citation
If you find this project useful in your research, please consider citing:

```bibtex
@article{chen2024slidechat,
  title={SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding},
  author={Chen, Ying and Wang, Guoan and Ji, Yuanfeng and Li, Yanjun and Ye, Jin and Li, Tianbin and and Ming, Hu and Yu, Rongshan and Qiao, Yu and He, Junjun},
  journal={arXiv preprint arXiv:2410.11761},
  year={2024}
}