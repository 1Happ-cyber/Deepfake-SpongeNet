# SpongeNet for Deepfake Detection

This is the official implementation of the paper: **"SpongeNet: Preserving Forgery Traces by Knowledge Sponge with Binary Information Bottleneck for Deepfake Detection"**.

## ⚙️ Training Configuration & Hyperparameters

Because of space limitations in the main paper, this section documents the full set of training hyperparameters. They come from two places: the config file (`./configs/results_cifake_T.cfg`) and the fixed hyperparameters hard-coded in `./main_model.py` (optimizer, warm-up scheduler, BIB block, and the loss-weighting coefficients such as the `λ` referenced in the code comments).

### CheckPoint
Cross COCO_Fake : https://drive.google.com/file/d/1_40k3kGJrTvcDKEv492ZwcOF0dgKKhrw/view?usp=drive_link

Cross DFFD : https://drive.google.com/file/d/1bwWQ7eFx7wKzgzXvxi9xn6txejKjDsYC/view?usp=drive_link
