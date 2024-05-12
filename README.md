# GAN
Generative Adversarial Networks (GAN) model designed for classification - Semi-Supervised Generative Adversarial Networks (SSGAN)

The code is modified to make GAN behave as a classifier.
Thus the GAN model is used as a two class classifier here to distinguish between the class labels: fire and no-fire.

FIRE DETECTION:
Images were captured using 2 cameras- Lepton and MLX.
The thermal images obtained from thermal camera using the electronic eye designed are used to model a machine learning based fire detection system.
Around 600 thermal images are collected from each variant with 300 images having fire and 300 images without fire.
These  images were used as training and testing samples.

It was observed during data collection that the thermal image from MLX90640 camera displays the temperature at a particular pixel using a thermal map and the thermal image from Lepton camera maps the pixel with highest temperature in the entire scene to red color and pixel with least temperature to blue color and it is a RGB image. 
<img width="408" alt="image" src="https://github.com/Pranav01rb/GAN/assets/57988947/39859153-9566-4822-8475-90010d452f70">
