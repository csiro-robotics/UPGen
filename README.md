# UPGen: a Universal Plant Generator for bridging the species gap
This repository provides the steps and code to train and deploy deep learning models for image based plant phenotyping. It contains the dataset, models and inference code relating to our (Under-review) CVIU Paper: [Scalable learning for bridging the species gap in image-based plant phenotyping](https://arxiv.org/abs/2003.10757). For more information see our [webpage](https://csiro-robotics.github.io/UPGen_Webpage/).

# Scalable learning for bridging the species gap in image-based plant phenotyping
If you use the pretrained models, datasets or code presented here please cite:

### Paper
```
Ward, D. and Moghadam, P., 2020. Scalable learning for bridging the species gap in image-based plant phenotyping, arXiv preprint arXiv:2003.10757
```
### Paper (bibtex)
```
@article{ward2020scalable,
  title={Scalable learning for bridging the species gap in image-based plant phenotyping},
  author={Ward, Daniel and Moghadam, Peyman},
  journal={arXiv preprint arXiv:2003.10757},
  year={2020}
}
```

# The Model
For leaf instance segmentation we use the Matterport Mask R-CNN implementation.

## Pretrained Model
A pretrained model can be downloaded [here](https://cloudstor.aarnet.edu.au/plus/s/uT5G7Lu3tZ6SahS).
It was trained using UPGen data and the expected uses are to continue training with data of your particular plant species and imaging environment.
This is the approach we employed in [our paper](https://arxiv.org/abs/2003.10757) to achieve state of the art performance in the [CVPPP Leaf Segmentation Challenge](https://competitions.codalab.org/competitions/18405).

## Mask RCNN Framework (from the matterport readme)
If you use this code please cite the authors of the architecture implementation. 
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

# Setup
We suggest using the Annaconda package manager to install dependencies. In these instructions we provide two lists of dependencies, 'environment.txt' will work for every machine and 'environment-gpu.txt' will offer faster training and predictions by using a GPU. Use 'environment-gpu.txt' if you have a GPU in your computer.
1. Create a coda environment: ```conda create --name upgen --file environment.txt -c conda-forge```
2. Activate the environment: ```conda activate upgen```
3. Clone the Matterport Mask RCNN implementation: ```git clone https://github.com/matterport/Mask_RCNN.git```
4. Change directory into mask rcnn: ```cd Mask_RCNN```
5. Add Mask_RCNN to the PYTHON PATH: ```export PYTHONPATH=$PYTHONPATH:`pwd````

## Compatibility
This framework has been tested on Ubuntu 16.04.

# Training The Model
In this repository we provide a tutorial to train a leaf instance segmentation model using state of the art segmentation architecture, Mask-RCNN. Our synthetic data is applicable to a variety of additional image analysis tasks including leaf counting and plant segmentation amoungst [others](http://phenotiki.com/traits.html).

Training the model occurs in train_cvppp.py. To see the different options and parameters for running this run ```python train_cvppp.py --help```

## Preparing The Data
Any data used for training will need to be sorted into the following set format. Alternatively one can implement their own version of the dataset class (dataset_cvppp.py) for their own folder structure.
Each dataset is expected to contain 2 directories, 'train' and 'crossVal' which contains the training and crossvalidation data. Each image file has a unique identifier (UID) is named 'plant<UID>_rgb.png' and the corresponding leaf instance segmentation label is 'plant<UID>_label.png'. An example file tree is:
```
    Dataset-Directory
    ├── crossVal
    │   ├── plant00000_label.png
    │   ├── plant00000_rgb.png
    │   ├── plant00001_label.png
    │   ├── plant00001_rgb.png
    │   ├── plant00002_label.png
    │   ├── plant00002_rgb.png
    │   ├── .
    │   ├── .
    │   ├── .
    │   ├── plant00025_label.png
    │   └── plant00025_rgb.png
    └── train
        ├── plant00026_label.png
        ├── plant00026_rgb.png
        ├── plant00027_label.png
        ├── plant00027_rgb.png
        ├── .
        ├── .
        ├── .
        ├── plant00299_label.png
        └── plant00299_rgb.png
```

## Training From Scratch
To train from scratch and initialize the model with random weights use ```--init rand```.
For example: 
```python train_cvppp.py --dataDir /path/to/your/datasets/upgen/train --outputDir /path/to/your/training/directory --name TrainingFromScratchUpgen_1 --numEpochs 5 --blurImages --init rand```

## Training From Pretrained Weights
Our large synthetic dataset can replace large computer vision datasets such as ImageNet or MS-COCO as a starting point for training. Mask-RCNN weights pretrained on our synthetic data can be downloaded [here](https://cloudstor.aarnet.edu.au/plus/s/uT5G7Lu3tZ6SahS).

To initialise training from pretrained weights pass the path using: ```--init /path/to/weights/pretrainedModel.h5```
For example: 
```python train_cvppp.py --dataDir /path/to/your/datasets/upgen/train --outputDir /path/to/your/training/directory --name PretrainedUpgen_1 --numEpochs 5 --blurImages --init /path/to/weights/pretrainedModel.h5```

## Fine Tuning On A Custom Dataset
The same method of loading pretrained weights enables one to fine tune a model on their own dataset. 
1. Download pretrained weights or train a model on a dataset of your choosing using the steps above.
2. Package your dataset into the required format using the information above.
3. Train the model again, using passing the path to the pretrained model as above, using: ```--init /path/to/weights/pretrainedModel.h5```
For example: ```python train_cvppp.py --dataDir /path/to/your/datasets/cvppp/train --outputDir /path/to/your/training/directory --name FineTuneingCVPPP_1 --numEpochs 5 --blurImages --init /path/to/weights/pretrainedModel.h5```

# Evaluating The Model
In this section we provide the steps to evaluate a model on data with corresponding leaf instance segmentation labels. As with training, we expect the image, label pair's filenames to be in the format 'plant<UID>_rgb.png' and 'plant<UID>_label.png.

To evaluate a model use 'evaluate.py'. To see all options run ```python evaluate.py --help```. This script will evaluate each image in the 'dataDir' and output the following into 'outputDir':
* '*_rgb.png': For each image a figure which compares predicted and ground truth annotations
* 'overallResults.csv': The following segmentation metrics (mean and standard deviation) for all images: 
  * Symmetric best dice (Dice)
  * mAP
  * difference in count between ground truth and predicted leaf count (DiC) and;
  * absolute difference in count between ground truth and predicted leaf count (|DiC|)
* 'sampleResults.csv': The same metrics as above for each individual sample evaulated.


An example running this command: ```python evaluate.py --dataDir --dataDir /path/to/your/datasets/cvppp/test --outputDir /path/to/your/test/results/directory --weightsPath /path/to/your/training/directory/MyBestModel/checkpoints/cvppp20200421T1103/mask_rcnn_cvppp_0005.h5 --savePredictions```

# Deploying The Model
In this section we provide steps to perform inference and obtain predictions for images. This does NOT require ground truth annotations or the data to be organised in a specific directory structure. Instead images are identified by a glob file pattern and standard special characters. For example 'myDir/*.png' will use all images with a filename ending in '.png'. 

To evaluate a model use 'inference.py'. To see all options run ```python inference.py --help```. This script will evaluate each image and output the following into 'outputDir':
* The prediction for each image using the same filename as the input file and
* A file called 'leafCounts.csv' which contains the predicted number of leaves for each image.

An example running this command: ```python inference.py --dataPattern '/path/to/your/datasets/InTheWild/test/*_rgb.png' --outputDir /path/to/your/results/directory --weightsPath /path/to/your/training/directory/MyBestModel/checkpoints/cvppp20200421T1103/mask_rcnn_cvppp_0005.h5```





