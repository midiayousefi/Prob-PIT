# Probabilistic Permutation Invariant Training (Prob-PIT)
Prob-PIT [1] is an extended version of Permutation Invariant Training (PIT)[2]. PIT trains a neural network that separates the speaker-specific speech signals, and then determines the best output-label assignment which minimizes the separation error. Finding the best output-label assignment has been a challenge in speech separation, which is referred to as label permutation ambiguity. PIT employs a hard decision to choose the output- label assignment. This approach is suboptimal, especially in the initial steps of training when the network generates unreliable outputs and the costs of different permutations are close. Prob-PIT considers the output-label permutation as a discrete latent random variable with a uniform prior dis- tribution. Prob-PIT defines a log-likelihood function based on the prior distributions and the separation errors of all possible permutations. Next, the network is trained by maximizing the log-likelihood function. Unlike the conventional PIT that uses one output-label permutation with the minimum cost, Prob-PIT uses all permutations by employing the soft-minimum function.
# system requirements

  - Python 2.7
  - Tensorflow 1.10.4
  - Kaldi
  
# How to run:
This repository contains the LSTM-based Network for speech separation with Prob-PIT loss function. The easiest way to run the entire separation system is to download the PIT separation system [here](https://github.com/pchao6/LSTM_PIT_Speech_Separation.git) and replace the model.py with the probabilistic_pit_model.py in this repository.

[1] Yousefi, Midia, Soheil Khorram, and John HL Hansen. "Probabilistic Permutation Invariant Training for Speech Separation." arXiv preprint arXiv:1908.01768 (2019).
[2]Yu, Dong, et al. "Permutation invariant training of deep models for speaker-independent multi-talker speech separation." 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
