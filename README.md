# DigitalImageProcessing
# Histogram Of Oriented Gradients Application for Pedestrian Detection 

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
```
Python 3.6
```
```
Numpy
```
```
Scikit-learn
```
```
Open-CV
```

### Project Directory Structure

    .
    ├── dataset                    
    │   ├── detect        #images for detection   
    │   ├── hardneg       #images for hard negative training  
    │   └── train         #positive training images
    │       ├──neg
    │       └──pos
    └── testdataset       #images for testing the performance of the classifier
    │       ├──neg
    │       └──pos
    │
    └──output              #consists of result images
### Usage
Clone the repository using the following command
```
git clone https://github.com/sourabhkumar0308/DigitalImageProcessing.git
```
```
cd DigitalImageProcessing
```
To run the Project there are two ways
```
1.) either use '.py' files
2.) or use 'ipynb' files (requires jupyter notebook)(recommended)
```
```
Run extract features.py or extract features.ipynb
```
```
Run detect.py or detect.ipynb
```
```
Run test.py or test.ipynb
```

```
person_final_hard.pkl is the pre-trained model. You can skip running 'extract features' if you are planning to use this model
```

