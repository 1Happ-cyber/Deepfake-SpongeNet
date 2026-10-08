# SpongeNet for Deepfake Detection

This is the official implementation of the paper: **"SpongeNet: Preserving Forgery Traces by Knowledge Sponge with Binary Information Bottleneck for Deepfake Detection"** which is accept by Pattern Recognition (PR) 
## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{CHEN2026114601,
  title = {SpongeNet: Preserving forgery traces by knowledge sponge with binary information bottleneck for deepfake detection},
  journal = {Pattern Recognition},
  pages = {114601},
  year = {2026},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2026.114601},
  url = {https://www.sciencedirect.com/science/article/pii/S0031320326015657},
  author = {Kai Chen and Qiming Wang and Zhuoyue Qin and Youwei Wang and Shizhe Hu}
}
```

## ⚙️ Training Configuration & Hyperparameters

Because of space limitations in the main paper, this section documents the full set of training hyperparameters. They come from two places: the config file (`./configs/results_cifake_T.cfg`) and the fixed hyperparameters hard-coded in `./main_model.py` (optimizer, warm-up scheduler, BIB block, and the loss-weighting coefficients such as the `λ` referenced in the code comments).

### CheckPoint
Cross COCO_Fake : https://drive.google.com/file/d/1_40k3kGJrTvcDKEv492ZwcOF0dgKKhrw/view?usp=drive_link

Cross DFFD : https://drive.google.com/file/d/1bwWQ7eFx7wKzgzXvxi9xn6txejKjDsYC/view?usp=drive_link

## Datasets
Our experiments and evaluations involve the following publicly available datasets:

### CIFAKE
- **Paper**: CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images
- **Link**: [https://arxiv.org/abs/2303.14126](https://arxiv.org/abs/2303.14126)

### DFFD (Diverse Fake Face Dataset)
- **Paper**: On the Detection of Digital Face Manipulation (CVPR 2020)
- **Link**: [https://arxiv.org/abs/1910.01717](https://arxiv.org/abs/1910.01717)

### COCOFake
- **Paper**: Parents and children: Distinguishing multimodal deepfakes from natural images
- **Link**: [https://arxiv.org/pdf/2304.00500](https://arxiv.org/pdf/2304.00500)

## Acknowledgements

Our code is built upon the excellent work of [fedeloper/binary_deepfake_detection](https://github.com/fedeloper/binary_deepfake_detection). A significant portion of our implementation is modified from this repository, and we sincerely thank the authors for open-sourcing their code.

## Note

Regarding **FLOPs calculation**: Standard FLOP counting tools (e.g., `FlopCountAnalysis`) treat binary convolution layers as regular `nn.Conv2d` layers, leading to inaccurate FLOPs estimation for binary neural networks.

To ensure a fair and accurate FLOPs report, we adopted the following strategy:

1. For the portion of the network inherited from [fedeloper/binary_deepfake_detection](https://github.com/fedeloper/binary_deepfake_detection), we directly use the FLOPs value reported in their original paper.
2. For the additional modules we introduced on top of the original architecture, we separately calculated their FLOPs.
3. The final reported FLOPs of our model is the **sum** of the above two parts.
