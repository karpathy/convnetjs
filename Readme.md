
# ConvNetJS

ConvNetJS implements Deep Learning models and learning algorithms as well as nice browser-based demos, all in Javascript.

For much more information, see the main page at:
http://cs.stanford.edu/people/karpathy/convnetjs/

## Online demos
- [Convolutional Neural Network on MNIST digits](http://cs.stanford.edu/~karpathy/convnetjs/demo/mnist.html)
- [Convolutional Neural Network on CIFAR-10](http://cs.stanford.edu/~karpathy/convnetjs/demo/cifar10.html)
- [Neural Network with 2 hidden layers on toy 2D data](http://cs.stanford.edu/~karpathy/convnetjs/demo/classify2d.html)

## Train your own models
To run these locally it is recommended that you use Nodejs or you may run into cross-origin security issues and not being able to load images. Chrome will have this problem. Firefox will work fine but I found Chrome to run much faster and more consistently.

To setup a nodejs server and start training:

1. install nodejs: `sudo apt-get install nodejs`
2. `cd` into convnetjs directory
3. install the connect library for nodejs to serve static pages `npm install connect`
4. `node nodejs_server.js`
5. Access the demos. http://localhost:8080/demo/classify2d.html will just work out of the box, but mnist.html and cifar10.html will require that you download the datasets and parse them into images. (You can also use the ones on my webserver if you're clever enough to see how to change the paths but naturally I'd prefer if you didn't use too much of my bandwidth). The python scripts I used to parse the datasets are linked to from the demo pages and require numpy and scipy.

If you'd like to use your own images, also don't miss the utility function `convnetjs.img_to_vol(document.getElementById('input_image'))` which takes an image element in the DOM as input and returns a convnetjs.Vol() ready to be consumed by ConvNetJS. Instead of loading images individually, you can also batch them up like I do for MNIST/CIFAR, with every image as a row in a large image. For example, one MNIST batch is a 10,000x768 image that I load once and then I pluck out a row at a time and reshape it into 28x28 image to use in a net.

If you're not working on images but have some custom data, you probably want just a basic neural network with no convolutions and pooling etc. You likely want to use the `FullyConnLayer` layer and stack it once or twice. Make sure to follow the FullyConnLayers with ReLU layers to introduce nonlinearities (append activation:'relu' in the layer definition), and also Dropout layers (append drop_prob:0.5 in layer definition, or any amount of dropout you desire).

## Example code
Import convnet.js into your document: `<script src="lib/convnet.js"></script>`

### For images
We first have to create a network. If you have images, here's an example network:

    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
    layer_defs.push({type:'conv', sx:5, filters:8, stride:1, activation:'relu'});
    layer_defs.push({type:'pool', sx:3, stride:2});
    layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
    layer_defs.push({type:'softmax', num_classes:10});

It takes 32x32x3 images (3 is for RGB), convolves with 8 5x5 filters with stride 1, uses Rectified Linear Unit activation function (i.e. it thresholds all values below zero to zero), then pools spatially, then there is a fully connected layer and finally a classifier. Again, don't miss convnetjs.img_to_vol() if you'd like to use your own images.

### Arbitrary non-image data : Classification
If you don't have images but some 2-D data, for example, your main building block is a FullyConnected layer:

    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons:40, activation:'relu', drop_prob:0.5});
    layer_defs.push({type:'softmax', num_classes:4});

Here we have a 2-layer neural network classifier for 4 classes working on 2-D points, where the second layer is also followed by dropout for regularization. The drop_prob must be in range (0,1).

To use a network or 2-D network,

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    var some_input = new convnetjs.Vol(1,1,2); // a 2-dimensional point
    var class_probabilities = net.forward(some_input); // forward props all layers in turn

To train the network we use the Trainer class:

    var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.9, batch_size:16, decay:0.001});
    for(var i=0;i<my_dataset.length;i++) {
      var x = new convnetjs.Vol(1,1,2,0.0); // a 1x1x2 volume initialized to 0's.
      x.w[0] = my_data[i][0]; // Vol.w is just a list, it holds your data
      x.w[1] = my_data[i][1];
      trainer.train(x, my_labels[i]);
    }

Once you train the network, simply use `net.forward(x)` for predictions.

### Regression
There is also an implementation of L2 loss that can be used for regression to arbitrary values. Instead of softmax, as a last layer just use 

    layer_defs.push({type:'regression', num_neurons:3});

In example above we'd be regressing to a single output that you must specify in the backward call as a list:

      trainer.train(x, [0.5, 1.2, -0.7]);


## Layers
Every layer takes a 3D volume (dimensions of WIDTH x HEIGHT x DEPTH) and transforms it into a different 3D volume using some set of internal parameters. Some layers (such as pooling, dropout) have no parameters. Currently available layers are:

1. `Convolutional Layer`: convolves input volume with local filters of given size, at given stride.
2. `Pooling Layer`: max-pools neighboring activations in 3D volume, keeping depth the same but reducing width and height of volume
3. `Softmax`: this is a classifier layer that should currently be last. It computes probabilities via dense connections to the input volume and dot product with class weights.
4. `Dropout Layer`: implements dropout to control overfitting. Can be used after layers that have very large number of nodes for regularization. You don't have to add this explicitly, simply use drop_prob:0.5 (or other amount) in a layer def to automatically add a Dropout layer right after it.
5. `Local Contrast Normalization Layer`: Creates local competition among neurons along depth at specific location, for all locations.
6. `Fully Connected Layer`: a number of neurons connected densely to input volume. They each compute dot product with the input
7. `ReluLayer`: creates the ReLU (Rectified Linear Unit) activation function. You don't have to add this explicitly, simply use activation:'relu' in a layer def to follow that layer with ReLU.
8. `RegressionLayer`: can be replaced with Softmax to do regression instead of classification.
9. `SigmoidLayer`: can be used as nonlinearity instead of ReluLayer, computes the sigmoid function x->1/(1+e^(-x)) You don't have to add this explicitly, simply use activation:'sigmoid' in a layer def to follow that layer with the Sigmoid.

If you're not dealing with images, the only layer that is of interest is the Fully Connected Layer, which you probably want to stack once or twice on top of your input. You also may consider using a Dropout layer in places where there are a lot of parameters to control overfitting (overfitting = your validation accuracy is much lower than the training accuracy).

If you're dealing with images, your networks should look similar to what you see in the demos.

## Use in Node
1. install: `npm install convnetjs`
2. `var convnetjs = require("convnetjs");`

## License
MIT
