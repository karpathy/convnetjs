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
    var V2 = new V.constructor();
    V2.w = V.x.map((sx) => {
      return sx.map((sy) => {
        return sy.map((depth) => {
          return Math.tanh(depth);
        });
      });
    });
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    var V = this.in_act; // we need to set dw of this
    var V2 = this.out_act;
    V.dw = V2.w.map((sx, x) => {
      return sx.map((sy, y) => {
        return sy.map((depth, d) => {
          return (1.0 - depth * depth) * V2.dw[x][y][d];
        });
      });
    });
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type
    };
  }

}

export function fromJSON(json) {
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new TanhLayer({
    out_depth : json.out_depth,
    out_sx : json.out_sx,
    out_sy : json.out_sy,
  });
}