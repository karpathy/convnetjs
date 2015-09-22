// globals
var layer_defs, net, trainer;
var t = "\
layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:28, out_sy:28, out_depth:1});\n\
layer_defs.push({type:'fc', num_neurons:50, bias_pref: 1.0,  activation:'tanh'});\n\
layer_defs.push({type:'lstm', filters:20, bias_pref: 1.0});\n\
layer_defs.push({type:'fc', num_neurons:50, bias_pref: 1.0, activation:'step'});\n\
layer_defs.push({type:'fc', num_neurons:50, bias_pref: 1.0,  activation:'tanh'});\n\
layer_defs.push({type:'fc', num_neurons:50, bias_pref: 1.0,  activation:'tanh'});\n\
layer_defs.push({type:'regression', num_neurons:28*28});\n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.5, method:'adadelta', batch_size:1, l2_decay:0.01, l1_decay:0.0});\n\
";

// ------------------------
// BEGIN MNIST SPECIFIC STUFF
// ------------------------
var bi = 0;
var k = 0;
var sample_training_instance = function() {

  // find an unloaded batch
  bi = (bi + 1) % loaded_train_batches.length; //sequential
  
  //Math.floor(Math.random()*loaded_train_batches.length);
  var b = loaded_train_batches[bi];
  k = (k + 1) % 3000; // sample within the batch
  
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

  return {x:x, label:labels[n]};
}

var num_batches = 21; // 20 training batches, 1 test
var data_img_elts = new Array(num_batches);
var img_data = new Array(num_batches);
var loaded = new Array(num_batches);
var loaded_train_batches = [];

// int main
$(window).load(function() {

  $("#newnet").val(t);
  change_net();

  for(var k=0;k<loaded.length;k++) { loaded[k] = false; }

  load_data_batch(0); // async load train set batch 0 (6 total train batches)
  load_data_batch(20); // async load test set (batch 6)
  start_fun();
});

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

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;
var lix = 5;
var d0 = 0;
var d1 = 1;

// loads a training image and trains on it with the network
var paused = false;
var embed_samples = [];
var embed_imgs = [];
var load_and_step = function() {
  if(paused) return; 

  if(embed_samples.length === 0) { // happens once
    for(var k=0;k<200;k++) {
      var s = sample_training_instance();
      embed_samples.push(s);
      // render x and save it too
      //var I = render_act(s.x);
      //embed_imgs.push(I);
    }
  }
  var sample = sample_training_instance();
  step(sample); // process this image
}


var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var w2LossWindow = new cnnutil.Window(100);
var w1LossWindow = new cnnutil.Window(100);
var step_num = 0;
var colors = ["red", "blue", "green", "orange", "magenta", "cyan", "purple", "silver", "olive", "lime", "yellow"];
var step = function(sample) {
  // train on it with network
  var stats = trainer.train(sample.x, sample.x.w);
  
  if(isNaN(stats.cost_loss)){
    console.log('trainer returns NaN');
    console.log(sample);
    console.log(stats);
  }
  
  // keep track of stats such as the average training error and loss  
  xLossWindow.add(stats.cost_loss);
  w1LossWindow.add(stats.l1_decay_loss);
  w2LossWindow.add(stats.l2_decay_loss);

  // visualize training status
  var train_elt = document.getElementById("trainstats");
  train_elt.innerHTML = '';
  var t = 'Forward time per example: ' + stats.fwd_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  
  t = 'Backprop time per example: ' + stats.bwd_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  
  t = 'Regression loss: ' + f2t(xLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  
  t = 'L2 Weight decay loss: ' + f2t(w2LossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  
  t = 'L1 Weight decay loss: ' + f2t(w1LossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  
  t = 'Examples seen: ' + step_num;
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));

  // log progress to graph, (full loss)
  if(step_num % 200 === 0) {
    var xa = xLossWindow.get_average();
    var xw1 = w1LossWindow.get_average();
    var xw2 = w2LossWindow.get_average();
    if(xa >= 0 && xw1 >= 0 && xw2 >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
      lossGraph.add(step_num, xa);
      //lossGraph.add(step_num, xa + xw1 + xw2);
      lossGraph.drawSelf(document.getElementById("lossgraph"));
    }
  }

  step_num++;
}

// user settings 
var change_lr = function() {
  trainer.learning_rate = parseFloat(document.getElementById("lr_input").value);
  update_net_param_display();
}
var update_net_param_display = function() {
  document.getElementById('lr_input').value = trainer.learning_rate;
}
var toggle_pause = function() {
  paused = !paused;
  var btn = document.getElementById('buttontp');
  if(paused) { btn.value = 'resume' }
  else { btn.value = 'pause'; }
}
var dump_json = function() {
  document.getElementById("dumpjson").value = JSON.stringify(net.toJSON());
}
var clear_graph = function() {
  lossGraph = new cnnvis.Graph(); // reinit graph too 
}
var reset_all = function() {
  update_net_param_display();

  // reinit windows that keep track of val/train accuracies
  lossGraph = new cnnvis.Graph(); // reinit graph too
  step_num = 0;

  // enter buttons for layers
  var t = '';
  for(var i=1;i<net.layers.length-1;i++) { // ignore input and regression layers (first and last)
    var butid = "button" + i;
    t += "<input id=\""+butid+"\" value=\"" + net.layers[i].layer_type + "(" + net.layers[i].out_depth + ")" +"\" type=\"submit\" onclick=\"updateLix("+i+")\" style=\"width:80px; height: 30px; margin:5px;\";>";
  }
  $("#layer_ixes").html(t);
  $("#button"+lix).css('background-color', '#FFA');
}
var load_from_json = function() {
  var jsonString = document.getElementById("dumpjson").value;
  var json = JSON.parse(jsonString);
  net = new convnetjs.Net();
  net.fromJSON(json);
  reset_all();
}
var change_net = function() {
  eval($("#newnet").val());
  reset_all();
}