import Layer from "./layer.js";

export default class SVMLayer extends Layer{

  constructor(opt = {}){
    super(opt);
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'svm';
  }

  backward(y, use_webgl = false, is_training = false) {

    // compute and accumulate gradient wrt weights and bias of this layer
    let x = new Float64Array(storage(this.in_act.dw).buffer);
    let w = new Float64Array(storage(this.in_act.w).buffer);

    // we're using structured loss here, which means that the score
    // of the ground truth should be higher than the score of any other 
    // class, by a margin
    var yscore = w[y]; // score of ground truth
    var margin = 1.0;
    var loss = 0.0;
    for(var i = 0; i < this.out_depth; i++) {
      if(y === i) { 
        x[i] = 0;
      }else{
        var ydiff = -yscore + w[i] + margin;
        if(ydiff > 0) {
          // violating dimension, apply loss
          x[i] += 1;
          x[y] -= 1;
          loss += ydiff;
        }
      }
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