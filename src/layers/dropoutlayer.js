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
    this.dropped = new Float64Array(this.out_sx*this.out_sy*this.out_depth);
  }

  forward(V, is_training = false) {
    this.in_act = V;
    let V2 = new V.constructor();
    const [N0, N1, N2] = [V.w.length, V.w[0].length, V.w[0][0].length];
    if(is_training) {
      // do dropout
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
      let dp = SIMD.float32x4.splat(this.drop_prob);
      // scale the activations during prediction
      for(var x = 0; x < N0; x++) {
        for(var y = 0; y < N1; y++) {
          for(var d = 0; d < N2; d += 4) {
            let Vd = SIMD.float32x4(V2.w[x][y][d], V2.w[x][y][d+1], V2.w[x][y][d+2], V2.w[x][y][d+3]);
            Vd = SIMD.float32x4.mul(Vd, dp);
            V2.w[x][y][d] = Vd.x;
            V2.w[x][y][d+1] = Vd.y;
            V2.w[x][y][d+2] = Vd.z;
            V2.w[x][y][d+3] = Vd.w;
          }
        }
      }
    }
    this.out_act = V2;
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