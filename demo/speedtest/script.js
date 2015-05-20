function logEvent(str) {
  console.log(str);
  var d = document.createElement('div');
  d.innerHTML = str;
  document.getElementById('result').appendChild(d);
}

n = 0;
dtall = 0;

var vol = new convnetjs.VolType(128, 128, 3);

function runExample() {
  var t0 = +new Date();
  layer.forward(x);
  //layer.backward();
  var t1 = +new Date();
  var diff = t1 - t0;
  dtall += diff;
  n++;
  logEvent('ran example ' + n + ' in ' + diff + 'ms, estimated average = ' + (dtall / n).toFixed(3) + 'ms');
}

function start() {
  // Conv Layer definition used in convnet benchmarks
  layer = new convnetjs.ConvLayer({ in_sx:128, in_sy:128, in_depth:3, sx:11, filters:96, stride: 1, pad: 0});
  x = new vol();
  run1i = setInterval(runExample, 5); // start
}