
# ConvNetJS

ConvNetJS implements Deep Learning models and learning algorithms as well as nice browser-based demos, all in Javascript.

For much more information, see the main page at:
http://cs.stanford.edu/people/karpathy/convnetjs/

## Online demos
Convolutional Neural Network on MNIST digits: http://cs.stanford.edu/~karpathy/convnetjs/demo/mnist.html
Convolutional Neural Network on CIFAR-10: http://cs.stanford.edu/~karpathy/convnetjs/demo/cifar10.html
Neural Network with 2 hidden layers on toy 2D data: http://cs.stanford.edu/~karpathy/convnetjs/demo/classify2d.html

## Train your own models
To run these locally it is recommended that you use Nodejs or you may run into cross-origin security issues and not being able to load images. Chrome will have this problem. Firefox will work fine but I found Chrome to run much faster and more consistently.

To setup a nodejs server and start training:

1. install nodejs: `sudo apt-get install nodejs`
2. `cd` into convnetjs directory
3. install the connect library for nodejs to serve static pages `npm install connect`
4. `node nodejs_server.js`
5. Access the demos. http://localhost:8080/demo/classify2d.html will just work out of the box, but mnist.html and cifar10.html will require that you download the datasets and parse them into images. (You can also use the ones on my webserver if you're clever enough to see how to change the paths but naturally I'd prefer if you didn't use too much of my bandwidth). The python scripts I used to parse the datasets are linked to from the demo pages and require numpy and scipy. 

If you don't want to work on images but have some custom data, you probably want just a basic neural network with no convolutions and pooling etc. That means you probably want to use the `FullyConnLayer` layer and stack it once or twice. Right now only ReLU (Rectified Linear Units: i.e. x -> max(0,x)) nonlinearity is supported, more (Maxout, Sigmoid?) are coming soon perhaps (but ReLUs work very well in practice).

## Use in Node
1. install: `npm install convnetjs`
2. var convnetjs = require("convnetjs");

## License
MIT
