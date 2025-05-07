# Facial Verification using Synthetic Data

This is a facial verification pipeline loosely following FRCSynV2.

This repo has 2 sections:
- Preprocessing
- Training/Evaluation

**Preprocessing -> preprocess.py**

It takes in any dataset that's in the form of an ImageFolder, and turns it into an ImageFolder of pytorch tensors. The preprocessing that is done is facial alignment done with MTCNN, and normalization to the ImageNet mean and std. Preprocessing as tensors allows for easier access when training/evaluating.

**Training/Evaluation -> train.py**

It takes in the preprocessed ImageFolder for the training dataset and evaluation dataset, and then which loss function you want (Triplet loss by default). After every 5 Epochs, evaluation is done with the validation dataset, printing stats.

**STEPS TO RUN:**
1. Clone Facenet for MTCNN: ```git clone https://github.com/timesler/facenet-pytorch``` in root of project
2. Pip get requirements (modify depending on system) for both this project and Facenet: ```pip install -r requirements.txt``` 
3. Download the GANDiffFace dataset: https://github.com/PietroMelzi/GANDiffFace
4. Download a BUPT-CB dataset: https://buptzyb.github.io/CBFace/?reload=true
5. preprocessed both dataset ImageFolders with ```python preprocess.py PATH_TO_FOLDER```
6. Train model by either modifying code to point to preprocessed folders or add the correct args
```
python train.py

# or if you want ArcFace
python train.py --loss ArcFace
```
