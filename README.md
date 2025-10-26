# Vision Transformer (ViT) for Pet Classification

This project demonstrates two approaches to image classification using the Vision Transformer (ViT) architecture on the **Oxford-IIIT Pet Dataset**.

1.  **ViT from Scratch:** A complete implementation of the ViT model in PyTorch.
2.  **Transfer Learning:** Using a pretrained ViT (`vit_b_16`) from `torchvision` and fine-tuning it for the specific dataset.

-----

## 1\. ViT from Scratch

This part builds the ViT architecture from the ground up, as described in the paper "An Image is Worth 16x16 Words."

### Key Components:

  * **Patch Embedding:** A `nn.Conv2d` layer is used to convert 16x16 image patches into flattened linear embeddings.
  * **Class Token (CLS):** A special learnable token (`[cls]`) is prepended to the sequence of patch embeddings. The final output corresponding to this token is used for classification.
  * **Positional Embedding:** Learnable parameters are added to the patch embeddings to retain positional information.
  * **Transformer Encoder:** The core of the model consists of a series of standard Transformer Encoder blocks. Each block contains:
      * Multi-Head Self-Attention
      * A Multilayer Perceptron (MLP)
      * Layer Normalization and residual connections.

**Result:** Training this model from scratch on the small Pet dataset resulted in very low accuracy (\~3-4%), demonstrating that ViT models typically require large-scale pretraining to perform well.

-----

## 2\. Transfer Learning with Pretrained ViT

This approach leverages a model pretrained on the large ImageNet dataset to achieve high accuracy.

### Process:

  * **Load Pretrained Model:** A `vit_b_16` model with default weights is loaded from `torchvision.models`.
  * **Freeze Backbone:** All parameters in the pretrained model are frozen (`requires_grad=False`) to prevent them from being updated during training.
  * **Replace Classifier Head:** The final classification head (`model.heads`) is replaced with a new `nn.Linear` layer, with an output size matching the 37 classes of the Pet dataset. This new head is the *only* part of the model that is trained.
  * **Hugging Face Wrapper:** The model is wrapped in a custom class that includes a `PyTorchModelHubMixin` to prepare it for easy uploading to the Hugging Face Hub, saving the model configuration (like class labels) alongside the weights.

**Result:** By only fine-tuning the final layer, this model achieved high validation accuracy **(\>92%)** in just a few epochs.

-----

## Requirements

The project uses the following main libraries:

  * `torch`
  * `torchvision`
  * `torchinfo` (for model summaries)
  * `matplotlib` & `seaborn` (for plotting)
  * `scikit-learn` (for metrics like `accuracy_score` and `confusion_matrix`)
  * `huggingface_hub` (for the Hugging Face wrapper)

You can install them with:

```bash
pip install torch torchvision torchinfo matplotlib seaborn scikit-learn huggingface_hub
```
