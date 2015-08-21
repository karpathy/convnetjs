import Layer from "./layer.js";

// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.

export default class RegressionLayer extends Layer{

  constructor(opt = {}){
    super(opt);
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'regression';
  }

  // y is a list here of size num_inputs
  // or it can be a number if only one value is regressed
  // or it can be a struct {dim: i, val: x} where we only want to 
  // regress on dimension i and asking it to have value x
  backward(y) { 
    if(typeof y == 'number'){
      y = [y];
    }

    // compute and accumulate gradient wrt weights and bias of this layer
    // zero out the gradient of input Vol
    let x = new Float64Array(TypedObject.storage(new this.in_act.constructor({w:this.in_act.w})).buffer);
    let loss = 0.0;

    for(let i = 0; i < this.out_depth && i < y.length; i++){
      let dy = x[i] - y[i];
      x[(x.length/2)+i] = dy;
      loss += 0.5 * dy * dy;
    }

    return loss;
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