// globals
var layer_defs, net, trainer;
var t = "\
layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:67});\n\
layer_defs.push({type:'lstm', filters:256, bias_pref: 1.0, drop_prob: 0.8});\n\
layer_defs.push({type:'fc', num_neurons:256, bias_pref: 1.0, activation:'step', drop_prob: 0.8});\n\
layer_defs.push({type:'lstm', filters:256, bias_pref: 1.0, drop_prob: 0.8});\n\
layer_defs.push({type:'softmax', num_classes:67});\n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.5, method:'adadelta', batch_size:1, l2_decay:0.001, l1_decay:0.0});\n\
";

var literature_str;
var literature_length;
var char2Ind = {};
var ind2Char = {};
var indCnt;
var prevChar;
var curChar;
var trainIter = 0;

// int main
$(window).load(function() {
  start_fun();
});

var load_training = function(){
  literature_str = document.getElementById("training_data").value;
  literature_length = literature_str.length;
  
  indCnt = 0;
  for(var i = 0; i < literature_length; i++){
    var c = literature_str.charAt(i)
    if(isNaN(char2Ind[c])){
      char2Ind[c] = indCnt;
      ind2Char[indCnt] = c;
      indCnt++;
    }
  }
  
  var train_data_elt = document.getElementById("training_data_stats");
  train_data_elt.innerHTML = '';
  
  var t = 'Training Data Length: ' + literature_length;
  train_data_elt.appendChild(document.createTextNode(t));
  train_data_elt.appendChild(document.createElement('br'));
  
  t = 'Unique Classes: ' + indCnt;
  train_data_elt.appendChild(document.createTextNode(t));
  train_data_elt.appendChild(document.createElement('br'));
}

var start_fun = function() {
  load_training();
  
  $("#newnet").val(t);
  change_net();
  
  setInterval(load_and_step, 0); // lets go!
}

var output_length = 2000;
var output_test = function(){
  //TODO: implement test output + reset function
  net.reset(); 
  var outString = "";
  var prevInd = Math.floor(Math.random()*literature_length);
  var x = new convnetjs.Vol(1,1,indCnt,0.0);
  x.w[prevInd] = 1;
  
  for(var i = 0; i < output_length; i++){
    var act = net.forward(x) 
    var prevProp = 0;
    var randProp = Math.random();
    var maxInd = 0;
    for(var j = 0; j < act.w.length; j++){
      prevProp += act.w[j];
      if(prevProp >= randProp){
        maxInd = j;
        break;
      }
    }
    outString += ind2Char[maxInd];
    x.w[prevInd] = 0;
    x.w[maxInd] = 1;
    
    prevInd = maxInd;
  }
  
  document.getElementById("output_sequence").value = outString;
  net.reset(); //reset the recurrent net
}

// loads a training image and trains on it with the network
var paused = true;
var load_and_step = function() {
  if(paused) return; 
  
  prevChar = curChar;
  curChar = literature_str.charAt(trainIter);
  
  if(!prevChar){
    return;
  }
  
  var curInd = char2Ind[curChar];
  var prevInd = char2Ind[prevChar];
  
  var x = new convnetjs.Vol(1,1,indCnt,0.0);
  //var label = new convnetjs.Vol(1,1,indCnt,0.0);
  x.w[prevInd] = 1;
  //label.w[curInd] = 1;
  step({x : x, label: curInd}); // process this image
  
  trainIter = (trainIter + 1) % literature_length;
}


//=======================
// UI Output Codes
//=======================
var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var w2LossWindow = new cnnutil.Window(100);
var w1LossWindow = new cnnutil.Window(100);
var step_num = 0;
var colors = ["red", "blue", "green", "orange", "magenta", "cyan", "purple", "silver", "olive", "lime", "yellow"];
var step = function(sample) {
  // train on it with network
  var stats = trainer.train(sample.x, sample.label);
  
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
  if(step_num % 200 === 0) {
    var train_elt = document.getElementById("trainstats");
    train_elt.innerHTML = '';
    var t = 'Forward time per example: ' + stats.fwd_time + 'ms';
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    
    t = 'Backprop time per example: ' + stats.bwd_time + 'ms';
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    
    t = 'Cost loss: ' + f2t(xLossWindow.get_average());
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
  }
  // log progress to graph, (full loss)
  if(step_num % 1000 === 0) {
    var xa = xLossWindow.get_average();
    var xw1 = w1LossWindow.get_average();
    var xw2 = w2LossWindow.get_average();
    if(xa >= 0 && xw1 >= 0 && xw2 >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
      //lossGraph.add(step_num, xa);
      lossGraph.add(step_num, xa + xw1 + xw2);
      lossGraph.drawSelf(document.getElementById("lossgraph"));
    }
  }
  
  if(step_num % 2000 === 0) {
    output_test();
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
}
var load_from_json = function() {
  var jsonString = document.getElementById("dumpjson").value;
  var json = JSON.parse(jsonString);
  net = new convnetjs.Net();
  net.fromJSON(json);
  
  trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.5, method:'adadelta', batch_size:1, l2_decay:0.001, l1_decay:0.0});
  
  alert("net loaded");
  reset_all();
}
var change_net = function() {
  eval($("#newnet").val());
  reset_all();
}