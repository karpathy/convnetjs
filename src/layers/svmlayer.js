import * as Layer from "./layer.js";

export class SVMLayer extends Layer{

  constructor(opt = {}){
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'svm';
  }

  forward(V, is_training) {
    this.in_act = V;
    this.out_act = V; // nothing to do, output raw scores
    return V;
  }

  backward(y) {

    // compute and accumulate gradient wrt weights and bias of this layer
    let x = new Float64Array(TypedObject.storage(this.in_act.dw).buffer);
    let w = new Float64Array(TypedObject.storage(this.in_act.w).buffer);

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

  getParamsAndGrads() { 
    return [];
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
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new SVMLayer({
    out_depth : json.out_depth,
    out_sx : json.out_sx,
    out_sy : json.out_sy,
    num_inputs : json.num_inputs
  });
}