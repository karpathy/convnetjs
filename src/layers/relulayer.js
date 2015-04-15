import Vol from "./convnet_vol.js";

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)

export class ReluLayer {

  constructor(opts = {}){
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'relu';
  }

  forward(V, is_training) {
    this.in_act = V;
    var V2 = V.clone();
    var N = V.w.length;
    var V2w = V2.w;
    for(var i=0;i<N;i++) { 
      if(V2w[i] < 0) V2w[i] = 0; // threshold at 0
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
      if(V2.w[i] <= 0){ 
        V.dw[i] = 0; // threshold
      } else {
        V.dw[i] = V2.dw[i];
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
      layer_type : this.layer_type
    }
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
  }

}