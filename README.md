# neuralnet

**IN DEVELOPMENT** i.e. doesn't work at the moment

Implements a basic L-Layer neural network. Current features are

- Inputs are images only (classification is based on folder location)
- Binary classification only 
- Supports only relu and sigmoid activation functions


## How to use
Define the L layer net by using the `neuralnet.NewHyperParametersBuilder()` with the following options

- `AddLayers(a ActivationFuncName, neurons ...uint)` - adds layers with the specified neurons and activation function
- `AddNLayer(a ActivationFuncName, neurons uint, n uint)` - adds n indentical layers to the net
- `SetLearningRate(learningRate float64)` - The learning rate to use, defaults to 0.01
- `SetIterations(iterations uint)` - number of iterations used to train the model, defaults to 1000
- `SetInitFactor(initFactor float64)` - defines the noise factor used when initialising the bias (components are randomly set between 0 and `initFactor`). Defaults to 0.01

The last layer must be a single neuron using the `sigmoid` activation function for binary classification.

e.g.
```go
hyperParams, err := neuralnet.NewHyperParametersBuilder().
    AddLayers(neuralnet.ActivationFuncNameReLU, 2, 5, 4).
    AddLayer(neuralnet.ActivationFuncNameSigmoid, 1)
    SetLearningRate(0.05).
    SetIterations(2000).
    Build()
```

Once defined, create a training set using the `neuralnet.NewImageSetBuilder()` with the following options

- `WithPathPrefix(pathPrefix string)` - defines a root folder to use
- `AddFolder(pathToFolder string, classification bool)` - adds a folder to the training set, with the classification to use
- `AddImage(pathToImage string, classification bool)` - adds a single image, with the classification to use
- `ResizeImages(width, height uint)` - resize all the imagesa 

If the images are not being resized, they need to be all of the same height and width.

e.g.
```go
trainingDataSet, err := neuralnet.NewImageSetBuilder().
    WithPathPrefix("datasets/Vegetable Images/train").
    AddFolder("Cabbage", false).
    AddFolder("Carrot", true).
    ResizeImages(64, 64).
    Build()
```

More to come...