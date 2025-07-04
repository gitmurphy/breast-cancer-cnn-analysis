# Breast-Cancer-CNN-Analysis  
_An analysis of __pre-trained CNN architectures__ for detecting malignant vs. benign breast anomalies using __transfer learning__. This project evaluates the performance of __MobileNetV2, ResNet18__ and __VGG16__ models against the __VinDr-Mammo dataset__. Working towards a method of medical image analysis with limited resources will __improve accessibility.__ Following accuracy results that were not clinically viable, a __ResNet-50__ model pre-trained on medical-image data (RadImageNet) was subsequently included in the study._

![Project Grade](https://img.shields.io/badge/Project%20Grade%3A%20First%20Class-81%25-brightgreen)

[⬇️ Download full thesis (PDF, 1.43 MB)](docs/thesis.pdf)

---

## Launch Colab Notebooks

| Notebook (Model) | Open |
|------------------|---------------|
| **MobileNet V2 Investigation** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gitmurphy/breast-cancer-cnn-analysis/blob/main/mnv2_implementation_weightedrandomsampler.ipynb) |
| **ResNet-18 Investigation** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gitmurphy/breast-cancer-cnn-analysis/blob/main/resnet18_implementation_weightedrandomsampler_and_weightedloss.ipynb) |
| **VGG16 Investigation** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gitmurphy/breast-cancer-cnn-analysis/blob/main/vgg16_implementation_weightedrandomsampler_and_weightedloss.ipynb) |
| **ResNet-50 (RadImageNet)** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gitmurphy/breast-cancer-cnn-analysis/blob/main/resnet50_implementation_weightedrandomsampler.ipynb) |

## Results Summary  

| Model | Accuracy | Recall | F1-score | Notes |
|-------|----------|---------------------------|----------|-----------|
| **MobileNet V2** (ImageNet) | **0.78** | 0.48 | **0.34** | MobileNetV2 achieves the highest **accuracy and F1-score**, but it **exhibits the lowest recall, which is critical for a cancer detection model**. |
| **ResNet-18** (ImageNet) | 0.66 | **0.57** | 0.28 | Given this, it seems **more logical to proceed with ResNet**. However, **its performance remains relatively poor, indicating there is still significant room for optimisation**. |
| **VGG-16** (ImageNet) | 0.39 | 0.71 | 0.21 | Thesis confirms that **ImageNet VGG converges quickly but lacks precision**, leading to many false positives. |
| **ResNet-50** (RadImageNet) | 0.69 | 0.46 | 0.25 | Unlike ResNet18, the model **reacted badly to the application of both a weighted loss function and WeightedRandomSampler**. Training again without the weighted loss and using a standard BCE loss proved to suit the model better. |

### Final conclusion
This study evaluated the performance of pre-trained CNNs for classifying breast cancer using the VinDr-Mammo dataset. While the models did not achieve high accuracy, some key insights were gained:

- Class imbalance had a major impact, with networks tending to predict benign cases more often. Applying data-augmentation plus a WeightedRandomSampler reduced the impact of the imbalance on key performance metrics
- Working toward medical-image analysis with limited resources will improve accessibility but training deep CNNs on large datasets is slow and costly; reaching this stage cost more than €68.26 in Colab Pro GPU time.
- In head-to-head comparison, MobileNetV2 achieved the highest accuracy and F1-score, yet the lowest recall, whereas ResNet-18 was the most balanced performer, though still short of clinical expectations.
- A ResNet-50 backbone pre-trained on RadImageNet was fine-tuned, but did not achieve any improvements, scoring less than the original ResNet-18 under each metric.
- The findings of this study suggest that the pre-trained CNN backbones may not yet be sufficient for VinDr-Mammo classification; __off-the-shelf backbone models require domain-specific pre-training and further fine-tuning to reach clinically reliable recall.__
- Future work should explore more efficient training strategies, richer multi-faceted data, and architectures that pair CNN feature extractors (radiomics) with explainable models, so that early breast-cancer detection becomes both accurate and transparent.

## Related proof-of-concept

> **cnn-yoga-pose-detector**  
> <https://github.com/gitmurphy/cnn-yoga-pose-detector>

Using the same MobileNet V2 transfer-learning pipeline, this companion project classifies yoga poses and serves as a clear indication the model implementation strategy has the potential for adaptability across domains.
The model reached **83 % accuracy**, confirming that the implementation performs as expected on a more straightforward computer vision task.