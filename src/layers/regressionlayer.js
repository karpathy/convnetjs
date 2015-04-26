import * as Layer from "./layer.js";

// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.

export class RegressionLayer extends Layer{

  constructor(opt = {}){
    super(opt);
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'regression';
  }

  forward(V, is_training) {
    this.in_act = V;
    this.out_act = V;
    return V; // identity function
  }

  // y is a list here of size num_inputs
  // or it can be a number if only one value is regressed
  // or it can be a struct {dim: i, val: x} where we only want to 
  // regress on dimension i and asking it to have value x
  backward(y) { 

    // compute and accumulate gradient wrt weights and bias of this layer
    var x = new this.in_act.constructor({w:this.in_act.w}); // zero out the gradient of input Vol
    var loss = 0.0;
    if(y instanceof Array || y instanceof Float64Array) {
      for(var i=0;i<this.out_depth;i++) {
        var dy = x.w[i] - y[i];
        x.dw[i] = dy;
        loss += 0.5*dy*dy;
      }
    } else if(typeof y === 'number') {
      // lets hope that only one number is being regressed
      var dy = x.w[0] - y;
      x.dw[0] = dy;
      loss += 0.5*dy*dy;
    } else {
      // assume it is a struct with entries .dim and .val
      // and we pass gradient only along dimension dim to be equal to val
      var i = y.dim;
      var yi = y.val;
      var dy = x.w[i] - yi;
      x.dw[i] = dy;
      loss += 0.5*dy*dy;
    }
    return loss;
  }

  getParamsAndGrads() { 
    return new Float64Array();
  }

  toJSON() {
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      num_inputs : this.num_inputs
    };
  }

}

export function fromJSON(json) {
  if(typeof json == 'string'){
    json = JSON.parse(json);
  }
  return new RegressionLayer({
    out_depth : json.out_depth,
    out_sx : json.out_sx,
    out_sy : json.out_sy,
    layer_type : json.layer_type,
    num_inputs : json.num_inputs
  });
}