# Deep Neural Networks - Machine Learning Course Assignment

## Part 1: ANN from Scratch
Based on chapter 11 (ch11) of “Implementing a Multi-layer Artificial Neural Network from Scratch” of the book “Machine Learning with PyTorch and Scikit-Learn” by Raschka et al. (2022)

In the file **Part1_ANN_From_Scratch.ipynb**, I revised the code from ch11 so that the ANN built from scratch includes 2 hidden layers, instead of 1 hidden layer. 
The main work was done in the forward() and backwards() methods, as the gradients of the losses needed to be calculated using the "Chain Rule" to backpropagate for the training.
In addition, the prediction performance of the resulted ANN was evaluated using macro AUC on an ANN model built from scratch.

A comparison between the 2-hidden-layer ANN (Revised) results, the original code results (from the book, see ch11_OriginalCode_withAUC.ipynb) and the fully connected ANN as implemented in Keras (found in Part1_ANN_From_Scratch.ipynb), can be seen in Table 1 below. 

*Table 1 – Macro AUC comparison*
|      | Revised | Keras implementation |	Original code from the book |
| :---: | :---: | :---: | :---: |
| Test macro AUC |	0.993 |	0.982 |	0.991 |
| Validation Accuracy [%]	| 92.56 |	85.18 |	93.88 |

(all used batch size = 100, 20 epochs, MSE loss, learning rate = 0.1)


## Part 2: Pretrained CNNs
The code for this part is in the notebook: **Part2_Pretrained_CNNs.ipynb**.
In this part, I adapted 2 pretrained models, VGG19 and YOLOv5 to the Oxford 102 flowers.
The used dataset is the Oxford 102 category flower dataset. The dataset includes ~8000 images of flowers, of 102 flower categories.
The dataset was split to 50% Train, 25% Validation and 25% Test sets.

### VGG19 
I took the VGG19's pretrained model and froze its layers, then added a new classifier head as seen below
```Python
x = base_model_VGG19.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(102, activation='softmax')(x)
```

*VGG19's results*
<p align="center">
  <img alt="Light" src=https://github.com/IdanCGit/Machine-Learning-Course-Ex3/assets/139128502/b6a8a004-7a59-44e5-a5ec-50b417a7ff27 width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src=https://github.com/IdanCGit/Machine-Learning-Course-Ex3/assets/139128502/17b11c67-eed0-467f-9677-1a20aa4eb88f width="45%">
</p>

### YOLOv5
YOLOv5 is mainly an object detection model, but it also has classification models like: YOLOv5s-cls. 
To train the model on a custom dataset, such as the Oxford 102 flowers dataset in this task, I used the classify/train.py, and the testing was made using the classify/val.py.

*YOLOv5-cls' results*
<p align="center">
  <img alt="Light" src=https://github.com/IdanCGit/Machine-Learning-Course-Ex3/assets/139128502/e7fe0a1b-dcb8-4400-a9e9-4ba9e706f5f6 width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src=https://github.com/IdanCGit/Machine-Learning-Course-Ex3/assets/139128502/4e31f7ce-9a78-42c2-addf-d57fb2a77713 width="45%">
</p>

Though, the classification accuracy results were 72.3% using VGG19 with a new classifier head, and 98.1% (top 1) using YOLOv5s-cls model with the training script.
Both could use additional epochs, the YOLOv5s-cls model already has very high accuracy, but VGG19's loss and accuracy graph seem to will decrease more with additional epochs – though the train time will increase.



