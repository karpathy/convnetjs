import * as Layer from "./layer.js"

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x) 
// so the output is between -1 and 1.

export class TanhLayer extends Layer{

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
    let [N0, N1, N2] = [V.w.length, V.w[0].length, V.w[0][0].length]
    
    for(let x = 0; x < N0; x++){
      for(let y = 0; y < N1; y++){
        for(let d = 0; d < N2; d++){
          V2.w[x][y][d] = +(Math.tanh(V.w[x][y][d]));
        }
      }
    }

    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    let [N0, N1, N2] = [this.out_act.w.length, this.out_act.w[0].length, this.out_act.w[0][0].length]
    let ones = SIMD.float32x4.splat(1.0);
    for(let x = 0; x < N0; x++){
      for(let y = 0; y < N1; y++){
        for(let d = 0; d < N2; d += 4){
          let out = SIMD.float32x4(this.out_act.dw[x][y][d], this.out_act.dw[x][y][d+1], this.out_act.dw[x][y][d+2], this.out_act.dw[x][y][d+3]);
          let res = SIMD.float32x4.mul(SIMD.float32x4.sub(ones, SIMD.float32x4.mul(out, out)), out);
          this.in_act.dw[x][y][d] = res.x;
          this.in_act.dw[x][y][d+1] = res.y;
          this.in_act.dw[x][y][d+2] = res.z;
          this.in_act.dw[x][y][d+3] = res.w;
        }
      }
    }
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