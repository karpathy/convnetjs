import * as Layer from "./layer.js";

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)

export class ReluLayer extends Layer {

  constructor(opts = {}){
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'relu';
  }

  forward(V, is_training) {
    this.in_act = V;
    var V2 = new V.constructor(V);
    V2.w = V2.w.map((sx, x) => {
      return sx.map((sy, y) => { 
        return sy.map((depth, d) => {
          if(depth < 0){ 
            return 0; // threshold at 0
          } else {
            return depth;
          }
        });
      });
    });
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    var V = this.in_act; // we need to set dw of this
    var V2 = this.out_act;
    // zero out gradient wrt data
    V.dw = V.dw.map((sx, x) => {
      return x.map((sy, y) => {
        return y.map((depth, d) => {
          if(V2.w[x][y][d] <= 0){ 
            return 0; // threshold
          } else {
            return V2.dw[x][y][d];
          }
        })
      })
    });
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

}

export function fromJSON(json) {
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new ReluLayer({
    out_depth : json.out_depth,
    out_sx : json.out_depth,
    out_sy : json.out_sy
  });
}