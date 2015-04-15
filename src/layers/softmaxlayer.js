import Vol from "./convnet_vol.js";

// Layers that implement a loss. Currently these are the layers that 
// can initiate a backward() pass. In future we probably want a more 
// flexible system that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)

export class SoftmaxLayer {

  constructor(opt = {}){
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'softmax';
  }

  forward(V, is_training) {
    this.in_act = V;

    var A = new Vol(1, 1, this.out_depth, 0.0);

    // compute max activation
    var as = V.w;
    var amax = V.w[0];
    for(var i=1;i<this.out_depth;i++) {
      if(as[i] > amax) amax = as[i];
    }

    // compute exponentials (carefully to not blow up)
    var es = global.zeros(this.out_depth);
    var esum = 0.0;
    for(var i=0;i<this.out_depth;i++) {
      var e = Math.exp(as[i] - amax);
      esum += e;
      es[i] = e;
    }

    // normalize and output to sum to one
    for(var i=0;i<this.out_depth;i++) {
      es[i] /= esum;
      A.w[i] = es[i];
    }

    this.es = es; // save these for backprop
    this.out_act = A;
    return this.out_act;
  }

  backward(y) {

    // compute and accumulate gradient wrt weights and bias of this layer
    var x = this.in_act;
    x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

    for(var i=0;i<this.out_depth;i++) {
      var indicator = i === y ? 1.0 : 0.0;
      var mul = -(indicator - this.es[i]);
      x.dw[i] = mul;
    }

    // loss is the class negative log likelihood
    return -Math.log(this.es[y]);
  }

  getParamsAndGrads() { 
    return [];
  }

  toJSON() {
    var json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.num_inputs = this.num_inputs;
    return json;
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.num_inputs = json.num_inputs;
  }

}