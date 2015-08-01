
var t = "\n\
// lets use an example fully-connected 2-layer ReLU net\n\
var layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});\n\
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});\n\
layer_defs.push({type:'softmax', num_classes:10});\n\
\n\
// below fill out the trainer specs you wish to evaluate, and give them names for legend\n\
var LR = 0.01; // learning rate\n\
var BS = 8; // batch size\n\
var L2 = 0.001; // L2 weight decay\n\
nets = [];\n\
trainer_defs = [];\n\
trainer_defs.push({learning_rate:LR, method: 'sgd', momentum: 0.0, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'sgd', momentum: 0.9, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'adam', eps: 1e-8, beta1: 0.9, beta2: 0.99, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'adagrad', eps: 1e-6, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'windowgrad', eps: 1e-6, ro: 0.95, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:1.0, method: 'adadelta', eps: 1e-6, ro:0.95, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'nesterov', momentum: 0.9, batch_size:BS, l2_decay:L2});\n\
\n\
// names for all trainers above\n\
legend = ['sgd', 'sgd+momentum', 'adam', 'adagrad', 'windowgrad', 'adadelta', 'nesterov'];\n\
"

// ------------------------
// BEGIN MNIST SPECIFIC STUFF
// ------------------------
classes_txt = ['0','1','2','3','4','5','6','7','8','9'];

var use_validation_data = false;
var sample_training_instance = function() {

  // find an unloaded batch
  var bi = Math.floor(Math.random()*loaded_train_batches.length);
  var b = loaded_train_batches[bi];
  var k = Math.floor(Math.random()*3000); // sample within the batch
  var n = b*3000+k;

  // load more batches over time
  if(step_num%5000===0 && step_num>0) {
    for(var i=0;i<num_batches;i++) {
      if(!loaded[i]) {
        // load it
        load_data_batch(i);
        break; // okay for now
      }
    }
  }

  // fetch the appropriate row of the training image and reshape into a Vol
  var p = img_data[b].data;
  var x = new convnetjs.Vol(28,28,1,0.0);
  var W = 28*28;
  for(var i=0;i<W;i++) {
    var ix = ((W * k) + i) * 4;
    x.w[i] = p[ix]/255.0;
  }
  x = convnetjs.augment(x, 24);

  var isval = use_validation_data && n%10===0 ? true : false;
  return {x:x, label:labels[n], isval:isval};
}

var sample_test_instance = function() {
  var b = 20;
  var k = Math.floor(Math.random()*3000);
  var n = b*3000+k;

  var p = img_data[b].data;
  var x = new convnetjs.Vol(28,28,1,0.0);
  var W = 28*28;
  for(var i=0;i<W;i++) {
    var ix = ((W * k) + i) * 4;
    x.w[i] = p[ix]/255.0;
  }
  x = convnetjs.augment(x, 24);
  return {x:x, label:labels[n]};
}

var num_batches = 21; // 20 training batches, 1 test
var data_img_elts = new Array(num_batches);
var img_data = new Array(num_batches);
var loaded = new Array(num_batches);
var loaded_train_batches = [];
var step_num = 0;

// int main
var lossWindows = [];
var trainAccWindows = [];
var testAccWindows = [];
var lossGraph, trainGraph, testGraph;
$(window).load(function() {

  $("#layerdef").val(t);

  for(var k=0;k<loaded.length;k++) { loaded[k] = false; }
  load_data_batch(0); // async load train set batch 0 (6 total train batches)
  load_data_batch(20); // async load test set (batch 6)
  start_fun();

  reload();
});

var reload = function() {
  
  eval($("#layerdef").val()); // fills in trainer_spects[] array, and layer_defs

  var N = trainer_defs.length;
  nets = [];
  trainers = [];
  for(var i=0;i<N;i++) {
    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    var trainer = new convnetjs.Trainer(net, trainer_defs[i]);
    nets.push(net); 
    trainers.push(trainer);
  }

  step_num = 0;

  lossWindows = [];
  trainAccWindows = [];
  testAccWindows = [];
  for(var i=0;i<N;i++) {
    lossWindows.push(new cnnutil.Window(800));
    trainAccWindows.push(new cnnutil.Window(800));
    testAccWindows.push(new cnnutil.Window(800));
  }
  lossGraph = new cnnvis.MultiGraph(legend);
  trainGraph = new cnnvis.MultiGraph(legend);
  testGraph = new cnnvis.MultiGraph(legend);
}

var start_fun = function() {
  if(loaded[0] && loaded[20]) { 
    console.log('starting!'); 
    setInterval(load_and_step, 0); // lets go!
  }
  else { setTimeout(start_fun, 200); } // keep checking
}

var load_data_batch = function(batch_num) {
  // Load the dataset with JS in background
  data_img_elts[batch_num] = new Image();
  var data_img_elt = data_img_elts[batch_num];
  data_img_elt.onload = function() { 
    var data_canvas = document.createElement('canvas');
    data_canvas.width = data_img_elt.width;
    data_canvas.height = data_img_elt.height;
    var data_ctx = data_canvas.getContext("2d");
    data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
    img_data[batch_num] = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
    loaded[batch_num] = true;
    if(batch_num < 20) { loaded_train_batches.push(batch_num); }
    console.log('finished loading data batch ' + batch_num);
  };
  data_img_elt.src = "mnist/mnist_batch_" + batch_num + ".png";
}

// ------------------------
// END MNIST SPECIFIC STUFF
// ------------------------

// main iterator function
var load_and_step = function() {
  step_num++;
  var sample = sample_training_instance();
  var test_sample = sample_test_instance();

  // train on all networks
  var N = nets.length;
  var losses = [];
  var trainacc = [];
  testacc = [];
  for(var i=0;i<N;i++) {

    // train on training example
    var stats = trainers[i].train(sample.x, sample.label);
    var yhat = nets[i].getPrediction();
    trainAccWindows[i].add(yhat === sample.label ? 1.0 : 0.0);
    lossWindows[i].add(stats.loss);

    // evaluate a test example
    nets[i].forward(test_sample.x);
    var yhat_test = nets[i].getPrediction();
    testAccWindows[i].add(yhat_test === test_sample.label ? 1.0 : 0.0);

    // every 100 iterations also draw
    if(step_num % 100 === 0) {
      losses.push(lossWindows[i].get_average());
      trainacc.push(trainAccWindows[i].get_average());
      testacc.push(testAccWindows[i].get_average());
    }
  }
  if(step_num % 100 === 0) {
    lossGraph.add(step_num, losses);
    lossGraph.drawSelf(document.getElementById("lossgraph"));

    trainGraph.add(step_num, trainacc);
    trainGraph.drawSelf(document.getElementById("trainaccgraph"));

    testGraph.add(step_num, testacc);
    testGraph.drawSelf(document.getElementById("testaccgraph"));
  }
}
