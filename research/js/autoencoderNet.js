// globals
var net;
var t = "\
net = new convnetjs.netprototype.AutoEncoder({in_depth: 8});\n\
";

var literature_str;
var literature_length;
var char2Ind = {};
var ind2Char = {};
var indCnt; //unique classes
var trainPicker = 0;
var maxCharCode = 0;
var maxInputDimension = 8; //256 classes

var graphInterval = 1000;
var outputInterval = 200;

var convnetjs;

// int main
$(window).load(function() {
  start_fun();
});

var load_training = function(){
  literature_str = document.getElementById("training_data").value;
  literature_length = literature_str.length;
  
  indCnt = 0;
  maxCharCode = 0;
  char2Ind = {};
  ind2Char = {};
  for(var i = 0; i < literature_length; i++){
    var cCode = literature_str.charCodeAt(i);
    if(maxCharCode < cCode){
      maxCharCode = cCode;
    }
    if(isNaN(char2Ind[cCode])){
      char2Ind[cCode] = indCnt;
      ind2Char[indCnt] = cCode;
      indCnt++;
    }
  }
  
  var train_data_elt = document.getElementById("training_data_stats");
  train_data_elt.innerHTML = '';
  
  var t = 'Training Data Length: ' + literature_length;
  train_data_elt.appendChild(document.createTextNode(t));
  train_data_elt.appendChild(document.createElement('br'));
  
  t = 'Unique Classes: ' + indCnt + "; Max CharCode:" + maxCharCode;
  train_data_elt.appendChild(document.createTextNode(t));
  train_data_elt.appendChild(document.createElement('br'));
}

var start_fun = function() {
  load_training();
  
  $("#newnet").val(t);
  change_net();
  
  setInterval(load_and_step, 0); // lets go!
}

var output_test = function(){
  net.reset(); 
  var outString = "";
  
  var ind = Math.floor(Math.random()*indCnt);
  
  var inputCharCode = ind2Char[ind];
  
  var inputArr = int2binArray(inputCharCode,maxInputDimension);
  
  var x = new convnetjs.Vol(1,1,maxInputDimension,0.0);
  //x.w[inputCharCode] = 1;
  
  for(var i = 0; i < maxInputDimension; i++){
    x.w[i] = inputArr[i];
  }
  
  var act = net.forward(x);
  
  //var prediction = net.getPrediction();
  
  outString += "Input: " + inputCharCode + "\n";
  outString += "Input Raw: " + x.w.toString() + "\n";
  outString += "Output Raw: " + act.w.toString() + "\n";
  
  document.getElementById("output_sequence").value = outString;
  net.reset(); //reset the recurrent net
}

var getMaxInVol = function(act){
  var maxIter = 0;
  var maxVal = -99999999;
  for(var i = 0; i < act.w.length; i++){
    if(act.w[i] > maxVal){
      maxVal = act.w[i];
      maxIter = i;
    }
  }
  return maxIter;
}

// loads a training image and trains on it with the network
var paused = true;

//TODO debug this module
var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i]= 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
}

var int2binArray = function(intValue, arrLength){
    var binaryStr = (intValue >>> 0).toString(2);
    
    if(arrLength < binaryStr.length){ //should be assert
        return null;
    }
    var arr = zeros(arrLength);

    for(var i = 0; i < binaryStr.length; i++){
        var v = parseInt(binaryStr.charAt(i),2);
        arr[i] = v;
    }
    
    return arr;
};

var load_and_step = function() {
  if(paused) return;
  
  trainPicker = Math.round(Math.random()*indCnt) % indCnt;
  //trainPicker = (trainPicker + Math.random() * 100) % literature_length; //randomly pick the next inter
  var inputCharCode = literature_str.charCodeAt(trainPicker);
  
  var inputArr = int2binArray(inputCharCode,maxInputDimension);
  
  var x = new convnetjs.Vol(1,1,maxInputDimension,0.0);
  //x.w[inputCharCode] = 1;
  
  for(var i = 0; i < maxInputDimension; i++){
    x.w[i] = inputArr[i];
  }
  
  step({x : x, label: inputCharCode, encode: inputArr}); // process this image
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

var step = function(sample) {
  // train on it with network
  var stats = net.train(sample.x); //for regression
  //var stats = trainer.train(sample.x, sample.label); //for softmax
  
  if(isNaN(stats.loss)){
    // console.log('trainer returns NaN');
    // console.log(sample);
    // console.log(stats);
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
  if(step_num % graphInterval === 0) {
    var xa = xLossWindow.get_average();
    var xw1 = w1LossWindow.get_average();
    var xw2 = w2LossWindow.get_average();
    if(xa >= 0 && xw1 >= 0 && xw2 >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
      //lossGraph.add(step_num, xa);
      lossGraph.add(step_num, xa + xw1 + xw2);
      lossGraph.drawSelf(document.getElementById("lossgraph"));
    }
  }
  
  if(step_num % outputInterval === 0) {
    output_test();
  }

  step_num++;
}

// user settings 
var change_lr = function() {
  net.trainer.learning_rate = parseFloat(document.getElementById("lr_input").value);
  update_net_param_display();
}
var update_net_param_display = function() {
  document.getElementById('lr_input').value = net.trainer.learning_rate;
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
  
  net.trainer = new convnetjs.Trainer(net, {learning_rate:0.5, method:'adadelta', batch_size:1, l2_decay:0.00000001, l1_decay:0.0});
  
  alert("net loaded");
  reset_all();
}
var change_net = function() {
  convnetjs = window.convnetjs;
  
  eval($("#newnet").val());
  reset_all();
}