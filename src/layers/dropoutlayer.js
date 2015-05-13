import * as Layer from "./layer.js";
import * as Vol from "../structures/vol.js";

export default class DropoutLayer extends Layer {

  constructor(opt = {}){
    super(opt);
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'dropout';
    this.drop_prob = opt.drop_prob || 0.5;
    this.dropped = new Array(this.out_sx*this.out_sy*this.out_depth);
  }

  forward(V, use_webgl = false, is_training = false) {
    super.forward(V, use_webgl, is_training);
    this.out_act = new V.constructor(storage(this.in_act).buffer.slice(0));
    let v = new Float64Array(storage(this.in_act.w).buffer);
    let v2 = new Float64Array(storage(this.out_act.w).buffer);
    let dp = SIMD.float64x2.splat(this.drop_prob);
    if(is_training) {
      // do dropout
      for(let i = 0; i < v.length; i++){
        if(Math.random() < this.drop_prob) { 
          v2[i] = 0;
          this.dropped[i] = true; // drop! 
        } else {
          this.dropped[i] = false;
        }
      }
    } else {
      // scale the activations during prediction
      for(let i = 0; i < v.length; i += 2){
        SIMD.float64x2.store(v2, i, SIMD.float64x2.mul(dp, SIMD.float64x2.load(v, i)));
      }
    }
    return this.out_act; // dummy identity function for now
  }

  backward(use_webgl = false, is_training = false) {
    let v = new Float64Array(storage(this.in_act.dw).buffer); // we need to set dw of this
    let v2 = new Float64Array(storage(this.out_act.dw).buffer);
    for(let i = 0; i < v.length; i++){
      v[i] = (!(this.dropped)) ? v2[i] : 0; // copy over the gradient
    }
  }

  toJSON() {
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      drop_prob : this.drop_prob
    };
  }

}