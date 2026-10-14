---
permalink: /
title: ""
excerpt: "Gexin Huang — multimodal learning, visual reasoning, and generative models"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

## About

I am a Ph.D. student in Electrical and Computer Engineering at the [University of British Columbia](https://www.ubc.ca/), advised by [Dr. Xiaoxiao Li](https://tea.ece.ubc.ca/) in the [Trusted and Efficient AI (TEA) Lab](https://tea.ece.ubc.ca/). My research focuses on multimodal learning, vision–language reasoning, and generative models, with applications in computational pathology and medical imaging. I expect to graduate in December 2027.

Previously, I was a research intern at RIKEN and Shanghai AI Laboratory, a research assistant at Sun Yat-Sen University, and an algorithm engineer at Peng Cheng Laboratory. I received my M.S. in Pattern Recognition and Intelligent Systems from South China University of Technology.

## Research Focus

- **Evidence-grounded visual reasoning:** benchmarks and reusable evidence representations for reasoning over ultra-resolution images and whole-slide pathology images.
- **Multimodal computational pathology:** combining heterogeneous foundation models, biological knowledge, and image features for mutation prediction and survival analysis.
- **Controllable medical image generation:** diffusion models for interactive tumor editing and identity-preserving brain aging modeling.

<h2 id="research-industry-experience">Research &amp; Industry Experience</h2>

### University of British Columbia — Trusted and Efficient AI Lab

**Ph.D. Researcher** · Vancouver, Canada · September 2023–Present<br>
Advisor: Dr. Xiaoxiao Li

- **[UltraVR: Evidence-Grounded Visual Reasoning](https://arxiv.org/abs/2606.05576)**
  - Developed a benchmark of 490 questions across four ultra-resolution image domains, with structured annotations of intermediate visual facts and reasoning operations.
  - Evaluated 11 vision–language models and diagnosed grounding and local-perception bottlenecks, with downstream inference often recovering when intermediate visual facts were supplied.
- **[SlideBank: Whole-Slide Visual Reasoning](https://arxiv.org/abs/2609.00342)**
  - Designed concept-indexed evidence banks with spatial provenance, multi-scale retrieval, and persistent reuse across questions about whole-slide images.
  - Improved Patho-R1 WSI-VQA accuracy from 44.28% to 52.77% compared with thumbnail-only inference.
- **[LogitProd: Pathology Foundation-Model Fusion](https://arxiv.org/abs/2604.07779)**
  - Developed logit-level fusion for nine heterogeneous pathology foundation models without encoder retraining or feature alignment.
  - Achieved the best performance on 20 of 22 tasks spanning classification, mutation prediction, and survival analysis.
- **[Weakly Supervised HCC Survival Modeling](https://ssrn.com/abstract=5899092)**
  - Built a whole-slide-image survival risk pipeline using 324 resected hepatocellular carcinoma cases and integrated morphologic and molecular profiling.
  - Validated risk stratification in an independent cohort of 363 TCGA cases and characterized distinct prognostic signatures and a fibronodular subtype.

### RIKEN Center for Advanced Intelligence Project

**Research Intern** · Tokyo, Japan · May–November 2024<br>
Supervisor: Prof. Tatsuya Harada

- **[Interactive CT/MRI Tumor Editing](https://arxiv.org/abs/2503.06809)**
  - Developed sketch-conditioned latent diffusion with reference-image conditioning for interactive tumor editing across four public CT/MRI datasets.
  - Designed sketch refinement to repair imperfect user contours, combining structural guidance and reference-image features to control tumors while preserving surrounding anatomy.

### Shanghai AI Laboratory — Medical Image AI Group

**Research Intern** · Shanghai, China · December 2023–April 2024<br>
Supervisor: Prof. Shaoting Zhang

- **[Identity-Preserving Brain MRI Editing](https://arxiv.org/abs/2503.09634)**
  - Developed age- and identity-conditioned latent diffusion with longitudinal triplet supervision using cohorts of 1,639 adult and 1,272 infant MRI scans.
  - Combined source-image context, continuous age conditioning, and learned identity control to preserve individual anatomy during age transformations from a single scan.

### Sun Yat-Sen University — HCP Lab

**Research Assistant** · Guangzhou, China · June 2022–November 2023<br>
Advisor: Prof. Xiaodan Liang

- **[Multimodal Genetic Mutation Prediction](https://doi.org/10.1016/j.patcog.2026.113047)**
  - Designed knowledge graphs and graph-transformer label embeddings that integrate biological and linguistic gene priors.
  - Evaluated the method across nine TCGA and two CPTAC cancer types; published in *Pattern Recognition* (2026).

### Peng Cheng Laboratory — Institute of Visual Intelligence

**Algorithm Engineer** · Shenzhen, China · July 2021–May 2022

- Developed and trained a multimodal Transformer for histopathology image analysis on large-scale datasets.

### South China University of Technology — Brain–Computer Interface Center

**Graduate Researcher** · Guangzhou, China · January 2018–June 2021<br>
Advisors: Profs. Zhuliang Yu and Wei Wu

- **[EEG/MEG Source Imaging](https://pubmed.ncbi.nlm.nih.gov/36215381/):** Developed a data-synthesis-based spatiotemporal encoder–decoder and evaluated it on simulated and real EEG/MEG signals, resulting in a first-author *IEEE Transactions on Neural Networks and Learning Systems* publication.
- **Multimodal Brain Signals:** Built a Bayesian generative model for EEG–fMRI learning by combining neural networks and probabilistic graphical models.

### Xiamen Airlines

**Aircraft Electronic and Software Systems Engineer** · August 2016–November 2017

<h2 id="publications">Publications</h2>

1. **[Knowledge-Guided Multi-Modality Transformer for Multi-Label Genetic Mutation Prediction](https://doi.org/10.1016/j.patcog.2026.113047)**<br>
   **Gexin Huang**, Chenfei Wu, Mingjie Li, Xiaojun Chang, Ying Sun, Lei Xing, Xiaodan Liang, Liang Lin, Guang Yang, and Shen Zhao.<br>
   *Pattern Recognition*, 2026.

2. **[Electromagnetic Source Imaging via a Data-Synthesis-Based Convolutional Encoder–Decoder Network](https://pubmed.ncbi.nlm.nih.gov/36215381/)**<br>
   **Gexin Huang**, Ke Liu, Jiawen Liang, Chang Cai, Zheng Hui Gu, Feifei Qi, Yuanqing Li, Zhu Liang Yu, and Wei Wu.<br>
   *IEEE Transactions on Neural Networks and Learning Systems*, 2024.

## Preprints & Manuscripts

1. **[UltraVR: A Diagnostic Ultra-Resolution Image-VQA Benchmark for Evidence-Grounded Reasoning](https://arxiv.org/abs/2606.05576)**<br>
   **Gexin Huang**, Yanting Yang, Myeongkyun Kang, Beidi Zhao, Jun Zhou, Chen Zhou, Gang Wang, Zu-hua Gao, and Xiaoxiao Li.<br>
   2026 · Submitted to ICLR.

2. **[SlideBank: A Persistent Hierarchical Evidence Bank for Consistent Whole-Slide Reasoning](https://arxiv.org/abs/2609.00342)**<br>
   Beidi Zhao<sup>*</sup>, **Gexin Huang**<sup>*</sup>, Ciro Zhang<sup>*</sup>, Anqi Li, Yusheng Tan, Chen Zhou, Gang Wang, Zu-hua Gao, and Xiaoxiao Li.<br>
   2026 · Submitted to WACV. <sup>*</sup>Equal contribution.

3. **[Plug-and-Play Logit Fusion for Heterogeneous Pathology Foundation Models](https://arxiv.org/abs/2604.07779)**<br>
   **Gexin Huang**, Anqi Li, Yusheng Tan, Beidi Zhao, Gang Wang, Zu-hua Gao, and Xiaoxiao Li.<br>
   2026 · Submitted to WACV.

4. **[Artificial Intelligence-Driven Risk Stratification Reveals Distinct Prognostic Signatures and a Fibronodular Subtype of Hepatocellular Carcinoma](https://ssrn.com/abstract=5899092)**<br>
   **Gexin Huang**, Ali Bashashati, Katherine Rich, Xiaoxiao Li, and Zu-hua Gao.<br>
   2025 · Submitted to European Journal of Cancer.

5. **[Interactive Tumor Progression Modeling via Sketch-Based Image Editing](https://arxiv.org/abs/2503.06809)**<br>
   **Gexin Huang**, Ruinan Jin, Yucheng Tang, Can Zhao, Tatsuya Harada, Xiaoxiao Li, and Gu Lin.<br>
   2025 · Preprint.

6. **[See-in-Pairs: Reference Image-Guided Comparative Vision-Language Models for Medical Diagnosis](https://arxiv.org/abs/2506.18140)**<br>
   Ruinan Jin, **Gexin Huang**, Xinwei Shen, Qiong Zhang, Yan Shuo Tan, and Xiaoxiao Li.<br>
   2025 · Submitted to Medical Image Analysis.

7. **[Identity Preserving Latent Diffusion for Brain Aging Modeling](https://arxiv.org/abs/2503.09634)**<br>
   **Gexin Huang**, Zhangsihao Yang, Yalin Wang, Guido Gerig, Mengwei Ren, and Xiaoxiao Li.<br>
   2025 · Submitted to Medical Image Analysis.

## Education

- **University of British Columbia**, Vancouver, Canada<br>
  Ph.D. in Electrical and Computer Engineering · September 2023–December 2027 (expected).
- **South China University of Technology**, Guangzhou, China<br>
  M.S. in Pattern Recognition and Intelligent Systems · January 2018–June 2021.
- **Civil Aviation University of China**, Tianjin, China<br>
  B.S. in Automation · September 2012–June 2016.

## Honors & Awards

- President's Academic Excellence PhD Initial Award (2023).
- Excellence Prize of National Intelligent Car Race (2015).
- National College Students' Innovation Program — Excellence Project Completion (2014).
- Excellent Undergraduate Student (2015).
- First Prize Scholarships (2012–2013, 2020); Second Prize Scholarships (2014–2015, 2019).

## Academic Service

- **Reviewer:** Pattern Recognition, Medical Image Analysis, IEEE Transactions on Medical Imaging, and MICCAI (2025); CVPR (2023–2025); IEEE Transactions on Neural Networks and Learning Systems (2022–2023).
- **Teaching Assistant:** Advanced Techniques in Machine Learning; Digital Signal Processing (2019).
