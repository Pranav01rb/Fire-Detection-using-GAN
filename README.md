# GAN
Generative Adversarial Networks (GAN) model designed for classification - Semi-Supervised Generative Adversarial Networks (SSGAN)

The code is modified to make GAN behave as a classifier.
Thus the GAN model is used as a two class classifier here to distinguish between the class labels: fire and no-fire.

The project was a part of a ISRO Project at PES University, Banaglore, India.

Paper: https://ieeexplore.ieee.org/document/9993503

FIRE DETECTION:
Images were captured using 2 cameras- Lepton and MLX.
The thermal images obtained from thermal camera using the electronic eye designed are used to model a machine learning based fire detection system.
Around 600 thermal images are collected from each variant with 300 images having fire and 300 images without fire.
These  images were used as training validation and testing samples.
We were able to obtain a maximum accuracy of 100% using just 12 images for training. This is because of the ability of the generator to generate fake samples for training the model, which are almost same as real images, thus increasing the number of samples for training internally. 

It was observed during data collection that the thermal image from MLX90640 camera displays the temperature at a particular pixel using a thermal map and the thermal image from Lepton camera maps the pixel with highest temperature in the entire scene to red color and pixel with least temperature to blue color and it is a RGB image. 

MLX Camera images
<img width="408" alt="image" src="https://github.com/Pranav01rb/GAN/assets/57988947/39859153-9566-4822-8475-90010d452f70">
<img width="408" alt="image" src="https://github.com/Pranav01rb/GAN/assets/57988947/c237b1a9-e8e8-40eb-b301-62d5d7b3b482">

Lepton Camera images
<img width="463" alt="image" src="https://github.com/Pranav01rb/GAN/assets/57988947/cf20a23a-dd49-456f-a728-4cb25aaf83e8">

# Results
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
