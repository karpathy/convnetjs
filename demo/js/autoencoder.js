// globals
var layer_defs, net, trainer;
var t = "\
layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:28, out_sy:28, out_depth:1});\n\
layer_defs.push({type:'fc', num_neurons:50, activation:'tanh'});\n\
layer_defs.push({type:'fc', num_neurons:50, activation:'tanh'});\n\
layer_defs.push({type:'fc', num_neurons:2});\n\
layer_defs.push({type:'fc', num_neurons:50, activation:'tanh'});\n\
layer_defs.push({type:'fc', num_neurons:50, activation:'tanh'});\n\
layer_defs.push({type:'regression', num_neurons:28*28});\n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:1, method:'adadelta', batch_size:50, l2_decay:0.001, l1_decay:0.001});\n\
";

// ------------------------
// BEGIN MNIST SPECIFIC STUFF
// ------------------------
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

var render_act = function(A) {
  var w = A.w;
  var mm = maxmin(w);
  var s = 1;

  var canv = document.createElement('canvas');
  canv.className = 'rendera';
  var W = A.sx * s;
  var H = A.sy * s;
  canv.width = W;
  canv.height = H;
  var ctx = canv.getContext('2d');
  var g = ctx.createImageData(W, H);
  var d = 0;
  for(var x=0;x<A.sx;x++) {
    for(var y=0;y<A.sy;y++) {
      var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  
      for(var dx=0;dx<s;dx++) {
        for(var dy=0;dy<s;dy++) {
          var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
          for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
          g.data[pp+3] = 255; // alpha channel
        }
      }
    }
  }
  ctx.putImageData(g, 0, 0);
  return canv;
}

// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
var draw_activations = function(elt, A, scale, grads) {

  var s = scale || 2; // scale
  var draw_grads = false;
  if(typeof(grads) !== 'undefined') draw_grads = grads;

  // get max and min activation to scale the maps automatically
  var w = draw_grads ? A.dw : A.w;
  var mm = maxmin(w);

  // create the canvas elements, draw and add to DOM
  for(var d=0;d<A.depth;d++) {

    var canv = document.createElement('canvas');
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);

    for(var x=0;x<A.sx;x++) {
      for(var y=0;y<A.sy;y++) {
        if(draw_grads) {
          var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
        } else {
          var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  
        }
        for(var dx=0;dx<s;dx++) {
          for(var dy=0;dy<s;dy++) {
            var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
            for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
            g.data[pp+3] = 255; // alpha channel
          }
        }
      }
    }
    ctx.putImageData(g, 0, 0);
    elt.appendChild(canv);
  }  
}

var visualize_activations = function(net, elt) {

  // clear the element
  elt.innerHTML = "";

  // show activations in each layer
  var N = net.layers.length;
  for(var i=0;i<N;i++) {
    var L = net.layers[i];

    var layer_div = document.createElement('div');

    // visualize activations
    var activations_div = document.createElement('div');
    activations_div.appendChild(document.createTextNode('Activations:'));
    activations_div.appendChild(document.createElement('br'));
    activations_div.className = 'layer_act';
    var scale = 2;
    if(L.layer_type==='fc') scale = 10; // for softmax
    if(L.layer_type==='regression') {
      var Vvis = L.out_act.clone();
      Vvis.sx = 28; 
      Vvis.sy = 28;
      Vvis.depth = 1;
      draw_activations(activations_div, Vvis, scale);
    } else {
      draw_activations(activations_div, L.out_act, scale);

      if(i===0) {
        // also append the regression layer right nex tto input
        // so that it's easy to compare
        activations_div.appendChild(document.createElement('br'));
        activations_div.appendChild(document.createTextNode('Predicted reconstruction:'));
        activations_div.appendChild(document.createElement('br'));
        var Vvis = net.layers[net.layers.length-1].out_act.clone();
        Vvis.sx = 28; 
        Vvis.sy = 28;
        Vvis.depth = 1;
        draw_activations(activations_div, Vvis, scale);
      }
    }

    if(L.layer_type === 'fc' && i===1) {
      var filters_div = document.createElement('div');
      filters_div.appendChild(document.createTextNode('Weights:'));
      filters_div.appendChild(document.createElement('br'));
      for(var j=0;j<L.filters.length;j++) {
        var Lshow = L.filters[j].clone();
        Lshow.sx = 28; 
        Lshow.sy = 28; 
        Lshow.depth = 1;
        draw_activations(filters_div, Lshow, 2);
      }
      activations_div.appendChild(filters_div);
    }

    // visualize filters if they are of reasonable size
    if(L.layer_type === 'conv') {
      var filters_div = document.createElement('div');
      if(L.filters[0].sx>3) {
        // actual weights
        filters_div.appendChild(document.createTextNode('Weights:'));
        filters_div.appendChild(document.createElement('br'));
        for(var j=0;j<L.filters.length;j++) {
          draw_activations(filters_div, L.filters[j], 2);
        }
        // gradients
        filters_div.appendChild(document.createElement('br'));
        filters_div.appendChild(document.createTextNode('Gradients:'));
        filters_div.appendChild(document.createElement('br'));
        for(var j=0;j<L.filters.length;j++) {
          draw_activations(filters_div, L.filters[j], 2, true);
        }
      } else {
        filters_div.appendChild(document.createTextNode('Weights hidden, too small'));
      }
      activations_div.appendChild(filters_div);
    }
    layer_div.appendChild(activations_div);

    // print some stats on left of the layer
    layer_div.className = 'layer ' + 'lt' + L.layer_type;
    var title_div = document.createElement('div');
    title_div.className = 'ltitle'
    var t = L.layer_type + ' (' + L.out_sx + 'x' + L.out_sy + 'x' + L.out_depth + ')';
    title_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(title_div);

    if(L.layer_type==='conv') {
      var t = 'filter size ' + L.filters[0].sx + 'x' + L.filters[0].sy + 'x' + L.filters[0].depth + ', stride ' + L.stride;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }
    if(L.layer_type==='pool') {
      var t = 'pooling size ' + L.sx + 'x' + L.sy + ', stride ' + L.stride;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }

    // find min, max activations and display them
    var mma = maxmin(L.out_act.w);
    var t = 'max activation: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
    layer_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(document.createElement('br'));

    // number of parameters
    if(L.layer_type==='conv') {
      var tot_params = L.sx*L.sy*L.in_depth*L.filters.length + L.filters.length;
      var t = 'parameters: ' + L.filters.length + 'x' + L.sx + 'x' + L.sy + 'x' + L.in_depth + '+' + L.filters.length + ' = ' + tot_params;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }
    if(L.layer_type==='fc') {
      var tot_params = L.num_inputs*L.filters.length + L.filters.length;
      var t = 'parameters: ' + L.filters.length + 'x' + L.num_inputs + '+' + L.filters.length + ' = ' + tot_params;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }

    // css madness needed here...
    var clear = document.createElement('div');
    clear.className = 'clear';
    layer_div.appendChild(clear);

    elt.appendChild(layer_div);
  }
}

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
      var I = render_act(s.x);
      embed_imgs.push(I);
    }
  }
  var sample = sample_training_instance();
  step(sample); // process this image
}

var lix = 5;
var d0 = 0;
var d1 = 1;
function cycle() {
  var selected_layer = net.layers[lix];
  d0 += 1;
  d1 += 1;
  if(d1 >= selected_layer.out_depth) d1 = 0; // and wrap
  if(d0 >= selected_layer.out_depth) d0 = 0; // and wrap
  $("#cyclestatus").html('drawing neurons ' + d0 + ' and ' + d1 + ' of layer #' + lix + ' (' + net.layers[lix].layer_type + ')');
}
function updateLix(newlix) {
  $("#button"+lix).css('background-color', ''); // erase highlight
  lix = newlix;
  d0 = 0;
  d1 = 1; // reset these
  $("#button"+lix).css('background-color', '#FFA');

  $("#cyclestatus").html('drawing neurons ' + d0 + ' and ' + d1 + ' of layer with index ' + lix + ' (' + net.layers[lix].layer_type + ')');
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
  var t = 'Backprop time per example: ' + stats.bwd_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Regression loss: ' + f2t(xLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'L2 Weight decay loss: ' + f2t(w2LossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'L1 Weight decay loss: ' + f2t(w1LossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Examples seen: ' + step_num;
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));

  // visualize activations
  if(step_num % 100 === 0) {
    var vis_elt = document.getElementById("visnet");
    visualize_activations(net, vis_elt);
  }

  // visualize embedding
  if(step_num % 100 === 0) {

    var embcanvas = document.getElementById('embedding');
    var ctx = embcanvas.getContext("2d");
    var EW = embcanvas.width;
    var EH = embcanvas.height;

    // propagate a few training examples through the network and grab codes
    var xcodes = [];
    var ycodes = [];
    var ns = embed_samples.length; // number of samples
    for(var k=0;k<ns;k++) {
      var sample = embed_samples[k];
      net.forward(sample.x);
      var xcode = net.layers[lix].out_act.w[d0];
      var ycode = net.layers[lix].out_act.w[d1];
      xcodes.push(xcode);
      ycodes.push(ycode);
    }
    var mmx = cnnutil.maxmin(xcodes);
    var mmy = cnnutil.maxmin(ycodes);

    // draw every example into the canvas
    ctx.clearRect(0,0,EW,EH);
    for(var k=0;k<ns;k++) {
      var xpos = (EW-28*2)*(xcodes[k]-mmx.minv)/mmx.dv+28;
      var ypos = (EH-28*2)*(ycodes[k]-mmy.minv)/mmy.dv+28;
      // draw border according to class
      ctx.fillStyle = colors[embed_samples[k].label];
      ctx.fillRect(xpos-2,ypos-2,32,32);

      ctx.drawImage(embed_imgs[k], xpos, ypos );
    }

  }

  // log progress to graph, (full loss)
  if(step_num % 200 === 0) {
    var xa = xLossWindow.get_average();
    var xw1 = w1LossWindow.get_average();
    var xw2 = w2LossWindow.get_average();
    if(xa >= 0 && xw1 >= 0 && xw2 >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
      lossGraph.add(step_num, xa + xw1 + xw2);
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
  $("#cyclestatus").html('drawing neurons ' + d0 + ' and ' + d1 + ' of layer with index ' + lix + ' (' + net.layers[lix].layer_type + ')');

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
