# GAN
Generative Adversarial Networks (GAN) model designed for classification - Semi-Supervised Generative Adversarial Networks (SSGAN)

The code has been adapted to transform the GAN into a classifier. As such, the GAN model functions as a two-class classifier specifically designed to differentiate between two categories: "fire" and "no-fire."

This project was conducted as part of an ISRO-funded initiative at PES University, Bangalore, India.

Paper: https://ieeexplore.ieee.org/document/9993503

# Overview
This paper presents the development of an innovative fire detection system called the Electronic Eye (E-Eye), designed specifically for human space capsules. It explores two variants of the system, focusing on the integration of thermal imaging with advanced machine learning techniques to enhance detection capabilities.

# Methodology
FIRE DETECTION:
Images were captured using two thermal cameras: Lepton and MLX90640. The thermal images obtained from these cameras were integral in developing a machine learning-based fire detection system as part of the Electronic Eye design. In total, approximately 600 thermal images were collected from each camera variant, evenly split between images depicting fire and those without fire. These images served as the dataset for training, validation, and testing phases of the model.

Remarkably, the model achieved a maximum accuracy of 100% using only 12 images for training. This high efficiency can be attributed to the Generative Adversarial Network (GAN) used in the system. The GAN's generator was adept at creating synthetic yet realistic samples that augmented the training dataset, effectively enhancing the model's learning process without needing a large number of real images.

During the data collection phase, notable differences in image representation were observed between the two cameras. The MLX90640 camera displayed the temperature at specific pixels using a thermal map, which is critical for precise temperature monitoring. Conversely, the Lepton camera employed a color mapping technique where the highest temperatures in the scene were represented in red and the lowest in blue, providing a visually intuitive RGB image of thermal variations.

<img width="400" alt="image" src="https://github.com/Pranav01rb/GAN/assets/57988947/39859153-9566-4822-8475-90010d452f70">
<img width="400" alt="image" src="https://github.com/Pranav01rb/GAN/assets/57988947/c237b1a9-e8e8-40eb-b301-62d5d7b3b482">

Fig 1: MLX Camera Image

<img width="500" alt="image" src="https://github.com/Pranav01rb/GAN/assets/57988947/cf20a23a-dd49-456f-a728-4cb25aaf83e8">

Fig 2: Lepton Camera Image
# Results
One of the key results of the study was the machine learning model’s ability to reach a maximum recognition accuracy of 100%. This high level of accuracy was achieved using only 12 training images, showcasing the effectiveness of the Generative Adversarial Networks (GANs) used in the project. The GAN’s generator played a crucial role in enhancing the training dataset by generating synthetic images that realistically simulate actual fire scenarios. This capability allowed for comprehensive training with a limited set of actual images.

Another important result is the system's response time, which is around 80 milliseconds. This quick prediction time is essential for fire detection systems, especially in places like space capsules where fires can spread rapidly. The fast response of the Electronic Eye system means that potential fires can be identified and dealt with quickly, greatly improving safety.

Table 1: COMPUTATION TIME FOR GAN BASED FIRE DETECTION
| Description     | #Samples | #Epochs | MLX    | Lepton |
|-----------------|----------|---------|--------|--------|
| Training Time   | 12       | 200     | 15.96s | 15.70s |
| Training Time   | 12       | 500     | 38.22s | 38.26s |
| Training Time   | 200      | 200     | 24.29s | 24.35s |
| Training Time   | 200      | 500     | 61.75s | 58.18s |
| Prediction Time | 1        | -       | 82.2ms | 76.2ms |
# How to Cite
@INPROCEEDINGS{9993503,
  author={Monish Moger, B and Abhishek, H and Pranav, B and Roy, Pubali and Bharadwaj, Priya and Gupta, Akash and Sharma, Payal and Manikandan, J},
  booktitle={2022 IEEE International Conference on Aerospace Electronics and Remote Sensing Technology (ICARES)}, 
  title={Design and Evaluation of an Electronic Eye for Fire Detection in Human Space Capsule}, 
  year={2022},
  volume={},
  number={},
  pages={1-7}
# Reference
Bhattiprolu, S. (2023). python_for_microscopists. GitHub, “Semi-supervised GAN”, https://github.com/bnsreenu/python_for_microscopists/tree/master/259_semi_supervised_GAN
