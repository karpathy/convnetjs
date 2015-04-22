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

  forward(V, is_training = false) {
    this.in_act = V;
    this.out_act = new V.constructor();
    let v = new Float32Array(TypedObject.storage(this.in_act).buffer);
    let v2 = new Float32Array(TypedObject.storage(this.out_act).buffer);
      
    if(is_training) {
      // do dropout
      const [N0, N1, N2] = [V.w.length, V.w[0].length, V.w[0][0].length];
      for(var x = 0; x < N0; x++) {
        for(var y = 0; y < N1; y++) {
          for(var d = 0; d < N2; d++) {
            if(Math.random() < this.drop_prob) { 
              V2.w[x][y][d] = 0; 
              this.dropped[i] = true; // drop! 
            } else {
              this.dropped[i] = false;
            }
          }
        }
      }
    } else {
      // scale the activations during prediction
      for(let i = 0; i < v.length; i += 4){
        SIMD.float32x4.store(v2, i, SIMD.float32x4.mul(SIMD.float32x4.splat(this.drop_prob), SIMD.float32x4.load(v, i)));
      }
    }
    return this.out_act; // dummy identity function for now
  }

  backward() {
    let V = this.in_act; // we need to set dw of this
    let chain_grad = this.out_act;
    const [N0, N1, N2] = [V.w.length, V.w[0].length, V.w[0][0].length];
    for(var x = 0; x < N0; x++) {
      for(var y = 0; y < N1; y++) {
        for(var d = 0; d < N2; d++) {
          if(!(this.dropped[i])) { 
            V.dw[x][y][d] = chain_grad.dw[x][y][d]; // copy over the gradient
          }else{
            V.dw[x][y][d] = 0;
          }
        }
      }
    }
  }

  getParamsAndGrads() {
    return new Float64Array(0);
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

export function fromJSON(json){
  if(typeof json === "string"){
    json = JSON.parse(json);
  }
  return new DropoutLayer(json);
}