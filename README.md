# ImageClassifier
### My Python 2 uni project that was tested on a hidden dataset provided by the university, i managed to achieve 81% accuracy on that dataset
### This PyTorch implementation presents a Convolutional Neural Network inspired by the DenseNet style architecture. Its designed for image classification tasks and includes the following:

### <DenseBlock>: Implements the dense connectivity pattern where each layer is connected to every other layer in a feed-forward fashion. This promotes feature reuse and improves gradient flow.
### <TransitionLayer>: Reduces the spatial dimensions of feature maps between dense blocks, helping to control the model's complexity.
MyCNN: The main model class that combines multiple DenseBlocks and TransitionLayers.###

### Growthrate parameter controls the number of features added by each layer in a DenseBlock
### Flexible architecture allowing for easy adjustment of depth and width
### Efficient feature reuse through dense connections
### Batch normalization and ReLU activation for improved training stability and performance was suggested by the professor

#The model is suitable for various image classification tasks and can be easily changed to accomodate different input sizes and classes

#Model.pth is the already trained on the given university dataset



