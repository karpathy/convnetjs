import Vol from "./convnet_vol.js";

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x) 
// so the output is between -1 and 1.

export class TanhLayer {

  constructor(opt = {}){
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'tanh';
  }

  forward(V, is_training) {
    this.in_act = V;
    var V2 = V.cloneAndZero();
    var N = V.w.length;
    for(var i=0;i<N;i++) { 
      V2.w[i] = Math.tanh(V.w[i]);
    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    var V = this.in_act; // we need to set dw of this
    var V2 = this.out_act;
    var N = V.w.length;
    V.dw = global.zeros(N); // zero out gradient wrt data
    for(var i=0;i<N;i++) {
      var v2wi = V2.w[i];
      V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
    }
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
    return json;
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
  }

}