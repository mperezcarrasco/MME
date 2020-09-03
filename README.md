# MME

My attempt for reproducing [Semi-supervised Domain Adaptation via Minimax Entropy
](https://arxiv.org/abs/1904.06487), Saito K. et al. 2019.



# Results

**Office31**:

All the experiments were performed using PyTorch (v1.0.0) and the same data splits provided by the authors in their original implementation.


Alexnet (Accuracy):
|      | w->a (1-shot)| d->a (1-shot) | w->a (3-shot)| d->a (3-shot) |
|:------------- |:-------------:| :-------------:| :-------------:| :-------------:|
| [Original Implementation](https://github.com/VisionLearningGroup/SSDA_MME/) | 54.97% | 56.14% | 67.18% | - |
| This Implementation | - | 57.08% | 65.88% | 69.12% |
