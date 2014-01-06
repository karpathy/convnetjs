
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

If you don't want to work on images but have some custom data, you probably want just a basic neural network with no convolutions and pooling etc. That means you probably want to use the `FullyConnLayer` layer and stack it once or twice. Make sure to follow the FullyConnLayers with ReLU layers to introduce nonlinearities, or use activation:'relu' in the layer definition.

## Layers
Every layer takes a 3D volume (dimensions of WIDTH x HEIGHT x DEPTH) and transforms it into a different 3D volume using some set of internal parameters. Some layers (such as pooling, dropout) have no parameters. Currently available layers are:

1. `Convolutional Layer`: convolves input volume with local filters of given size, at given stride.
2. `Pooling Layer`: max-pools neighboring activations in 3D volume, keeping depth the same but reducing width and height of volume
3. `Softmax`: this is a classifier layer that should currently be last. It computes probabilities via dense connections to the input volume and dot product with class weights.
4. `Dropout Layer`: implements dropout to control overfitting. Can be used after layers that have very large number of nodes for regularization.
5. `Local Contrast Normalization Layer`: Creates local competition among neurons along depth at specific location, for all locations.
6. `Fully Connected Layer`: a number of neurons connected densely to input volume. They each compute dot product with the input
7. `ReluLayer`: creates the ReLU (Rectified Linear Unit) activation function.

If you're not dealing with images, the only layer that is of interest is the Fully Connected Layer, which you probably want to stack once or twice on top of your input. You also may consider using a Dropout layer in places where there are a lot of parameters to control overfitting (overfitting = your validation accuracy is much lower than the training accuracy).

If you're dealing with images, your networks should look similar to what you see in the demos.

## Use in Node
1. install: `npm install convnetjs`
2. `var convnetjs = require("convnetjs");`

## License
MIT
