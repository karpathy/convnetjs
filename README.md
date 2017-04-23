[![Build Status](https://travis-ci.org/yoskeoka/convnetjs-ts.svg?branch=master)](https://travis-ci.org/yoskeoka/convnetjs-ts)

# convnetjs-ts

This is a porting from ConvNetJS.
Now fully added type annotations and webpacked, so you can use this library in TypeScript for node.js and browser!
Of course you can use it in JavaScript.

## Install

Install
```
npm install convnetjs-ts
```

Import to your node project

JavaScript(ES6 or later/ TypeScript style)
```js
import * as convnetjs from "convnetjs-ts"; 
```

JavaScript(ES5 style)
```js
var convnetjs = require("convnetjs-ts");
```

for browser
build .js file is here:  `node_modules/convnetjs-ts/build/convnet.js`

but using webpack is recommended!

# ConvNetJS

ConvNetJS is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/Softmax) and Regression (L2) **cost functions**
- Ability to specify and train **Convolutional Networks** that process images
- An experimental **Reinforcement Learning** module, based on Deep Q Learning

For much more information, see the main page at [convnetjs.com](http://convnetjs.com)

## Example Code

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:

```js
import * as convnetjs from "convnetjs-ts"; 

// species a 2-layer neural network with one hidden layer of 20 neurons
var layer_defs = [];
// input layer declares size of input. here: 2-D data
// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
// then the first two dimensions (sx, sy) will always be kept at size 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'}); 
// declare the linear classifier on top of the previous hidden layer
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// forward a random data point through the network
var x = new convnetjs.Vol([0.3, -0.5]);
var prob = net.forward(x); 

// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101

var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
trainer.train(x, 0); // train the network, specifying that x is class zero

var prob2 = net.forward(x);
console.log('probability that x is class 0: ' + prob2.w[0]);
// now prints 0.50374, slightly higher than previous 0.50101: the networks
// weights have been adjusted by the Trainer to give a higher probability to
// the class we trained the network with (zero)
```

and here is a small **Convolutional Neural Network** if you wish to predict on images:  
TODO: convert function for Node.js

```js
import * as convnetjs from "convnetjs-ts"; 
var layer_defs = [];
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3}); // declare size of input
// output Vol is of size 32x32x3 here
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Vol of the same size
// output Vol will thus be 32x32x16 at this point
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 16x16x16 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 16x16x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 4x4x20 here
layer_defs.push({type:'softmax', num_classes:10});
// output Vol is of size 1x1x10 here

net = new convnetjs.Net();
net.makeLayers(layer_defs);

// helpful utility for converting images into Vols is included
var x = convnetjs.img_to_vol(document.getElementById('some_image'))
var output_probabilities_vol = net.forward(x)
```

and a very simple Reinforce-Learning smaple([This code refer to this sample](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html)):
```js
import { deepqlearn } from "convnetjs-ts";
const brainOpt = { start_learn_threshold: 100 };
const brain = new deepqlearn.Brain(3, 2, brainOpt); // 3 inputs, 2 possible outputs (0,1)
const state = [0, 0, 0];
for (let k = 0; k < 1000; k++) {
    const action = brain.forward(state); // returns index of chosen action
    const reward = action === 1 ? 1.0 : 0.0; //give a reward for action 1 (no matter what state is)
    brain.backward(reward); // <-- learning magic happens here
    state[Math.floor(Math.random() * 3)] = Math.random(); // change state
}
brain.epsilon_test_time = 0.0; // don't make any more random choices
brain.learning = false;
// get an optimal action from the learned policy
const input = [1, 1, 1];
const chosen_action = brain.forward(input);
console.log("chosen action after learning: " + chosen_action);
// tanh are their own layers. Softmax gets its own fully connected layer.
// this should all get desugared just fine.
```

## Getting Started
A [Getting Started](http://cs.stanford.edu/people/karpathy/convnetjs/started.html) tutorial is available on main page.

The full [Documentation](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html) can also be found there.

TODO: release convnetjs-ts  
See the **releases** page for this project to get the minified, compiled library, and a direct link to is also available below for convenience (but please host your own copy)

## License
MIT
