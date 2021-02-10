---
title: 'Reading List about Bayesian Deep Learning'
date: 2020-08-1
permalink: /posts/2020/08/reading-list/
tags:
  - machine learning
  - deep learning
  - Bayesian 
---

This is a reading collection about bayesian deep learning (BDL) and Deep Bayesian Learning (DBL). (last updated: 2020/10)


# Fundamental Books
* PRML Pattern Recognition and Machine Learning, Bishop 2006
* [Machine Learning: A Probabilistic Perspective](http://scholar.google.com.tw/scholar_url?url=https://research.google/pubs/pub38136.pdf&hl=zh-TW&sa=X&ei=PeAHYOa0HL3EywSi74rAAQ&scisig=AAGBfm0x2K8q8nf6AuaaZezSUpn5-CtgyA&nossl=1&oi=scholarr), Murphy 2012
* Bayesian Learning for Neural Networks, Neal 1996
* Deep learning, Goodfellow 2016
* PGM Probabilistic Graphical Models: Principles and Techniques, Koller and Friedman 2009

# Core
  ### core reseacrch areas
    * Bayesian Deep Learning in Approximation Inference
    * Representation Learning
    * Deep Genarative models
    * MCMC methods
  
---

## 1 Expectation Maximization (EM) and Variational Inference (VI):
* **PRML Chapter 9, 10.1-10.6**
* Variational Inference: A Review for Statisticians, Blei et al. 2016
* An Introduction to Variational Methods for Graphical Models, Jordan et al. 1999
Amortized Variational Inference and Reparameterization Trick:
* Auto-Encoding Variational Bayes, Kingma and Welling 2013
* Stochastic Backpropagation and Approximate Inference in Deep Generative Models, Rezende et al. 2014
* The Generalized Reparameterization Gradient, Ruiz et al. 2016
* Inference Suboptimality in Variational Autoencoders, Cremer et al. 2018
* Forward Amortized Inference for Likelihood-Free Variational Marginalization, Ambrogioni et al. 2018

### 1.1 Hierarchical Variational Methods:
* An Auxiliary Variational Method, Agakov and Barber 2004
* Hierarchical Variational Models, Ranganath et al. 2015
* Auxiliary Deep Generative Models, Maaløe et al. 2016
* Markov Chain Monte Carlo and Variational Inference: Bridging the Gap, Salimans et al. 2014
* Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
* The Variational Gaussian Process, Tran et al. 2015

### 1.2 Expectation Propagation (EP):
* PRML Chapter 10.7
* PGM Chapter 11.4
* Proofs of Alpha Divergence Properties (lecture note), Cevher 2008
* Divergence Measures and Message Passing, Minka 2005

### 1.3 Implicit Inference
* Adversarially Learned Inference, Dumoulin et al. 2016
* Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks, Mescheder et al. 2017
* Variational Inference using Implicit Distributions, Huszar 2017


## 2 Deep Generative Models (DGMs)

### Deep State Space Models (for tiem series)
* A Recurrent Latent Variable Model for Sequential Data, Chung et al. 2015
* Deep Kalman Filters, Krishnan et al. 2015
* Filtering Variational Objectives, Maddison et al. 2017
* Variational Sequential Monte Carlo, Naesseth et al. 2017
* Auto-Encoding Sequential Monte Carlo, Le et al. 2017
* Variational Bi-LSTMs, Shabanian et al. 2017

### 2.1 Variational Autoencoders 
* Importance Weighted Autoencoders, Burda et al. 2015
* Reinterpreting Importance-Weighted Autoencoders, Cremer et al. 2017
* Sequentialized Sampling Importance Resampling and Scalable IWAE, Huang and Courville 2018
* Tighter Variational Bounds are Not Necessarily Better, Rainforth et al. 2018
* On Nesting Monte Carlo Estimators, Rainforth et al. 2018
* Debiasing Evidence Approximations: On Importance-weighted Autoencoders and Jackknife Variational Inference, Nowozin 2018

### 2.2 Normalizing Flows
* Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
* Improving Variational Inference with Inverse Autoregressive Flow, Kingma et al. 2016
* Improving Variational Auto-Encoders using Householder Flow, Tomczak and Welling 2016
* Improving Variational Auto-Encoders using Convex Combination Linear Inverse Autoregressive Flow, Tomczak and Welling 2017
* Sylvester Normalizing Flows for Variational Inference, Berg et al. 2018
* Neural Autoregressive Flows, Huang et al. 2018
* Density Estimation using Real NVP, Dinh et al. 2016
* Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal 2018
* Neural Ordinary Differential Equations, Chen et al. 2018

### 2.3 Transfer Learning and Semisupervised Learning
* Semi-Supervised Learning with Deep Generative Models, Kingma et al. 2014
* Towards a Neural Statistician, Edwards and Storkey 2016
* One-Shot Generalization in Deep Generative Models, Rezende et al. 2016
* Uncertainty in Multitask Transfer Learning, Lacoste et al. 2018
* Conditional Neural Processes, Garnelo et al. 2018
* Neural Processes, Garnelo et al. 2018

### 2.4 Representation Learning
* Ladder Variational Autoencoders, Sønderby et al. 2016
* PixelVAE: A Latent Variable Model for Natural Images, Gulrajani et al. 2016
* Variational Lossy Autoencoder, Chen et al. 2016
* Generating Sentences from a Continuous Space, Bowman et al. 2015
* Generating Sentences by Editing Prototypes, Guu et al. 2017
* The Variational Fair Autoencoder, Louizos et al. 2015
* VAE with a VampPrior, Tomczak and Welling 2017
* Hierarchical VampPrior Variational Fair Auto-Encoder, Botros and Tomczak 2018
* Neural Relational Inference for Interacting Systems, Kipf et al. 2018
* Hyperspherical Variational Auto-Encoders, Davidson et al. 2018
* Neural Scene Representation and Rendering, Eslami et al. 2018

### 2.5 Bayesian Compression
* Bayesian Compression for Deep Learning, Louizos et al. 2017
* Improved Bayesian Compression, Federici et al. 2017
* Variational Dropout Sparsifies Deep Neural Networks, Molchanov et al. 2017
* Learning Sparse Neural Networks through L0 Regularization, Louizos et al. 2018
* Structured Variational Learning of Bayesian Neural Networks with Horseshoe Priors, Ghosh et al. 2018



## Bayesian Neural Networks 

### MCMC Approaches
* Bayesian Learning via Stochastic Gradient Langevin Dynamics, Welling and Teh 2011
* Bayesian Posterior Sampling via Stochastic Gradient Fisher Scoring, Ahn et al. 2012
* Stochastic Gradient Hamiltonian Monte Carlo, Chen et al. 2014
* Bayesian Sampling Using Stochastic Gradient Thermostats, Ding et al. 2014
* Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks, Li et al 2015
* Entropy-SGD: Biasing Gradient Descent Into Wide Valleys, Chaudhari et al. 2017
* Adversarial Distillation of Bayesian Neural Network Posteriors, Wang et al. 2018
Deep neural networks = Gaussian Process
* Priors for Infinite Network, Neal 1994
* Bayesian Learning for Neural Networks, Neal 1995
* Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, Gal and Ghahramani 2015
* Avoiding Pathologies in Very Deep Networks, Duvenaud et al. 2016
* Deep Neural Networks as Gaussian Processes, Lee et al. 2018
SGD / Approximate Inference / PAC-Bayes
* PAC-Bayesian Theory Meets Bayesian Inference, Germain et al. 2016
* Stochastic Gradient Descent as Approximate Bayesian Inference, Mandt et al. 2017
* Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles for Deep Networks, Chaudhari and Soatto 2017
* Generalization Bounds of SGLD for Non-convex Learning: Two Theoretical Viewpoints, Mou et al. 2017
* Entropy-SGD Optimizes the Prior of a PAC-Bayes Bound: Generalization properties of Entropy-SGD and data-dependent priors, Dziugaite and Roy 2017
* A Bayesian Perspective on Generalization and Stochastic Gradient Descent, Smith and Le 2018

### Inference Approaches 
