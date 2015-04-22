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
    let V2 = new V.constructor(V);

    let {sx, sy, depth} = V;

    let zeroes = SIMD.float32x4.splat(0.0);
    
    for(let x = 0; x < sx; x++){
      for(let y = 0; y < sy; y++){
        for(let d = 0; d < depth; d += 4){
          let vv2 = SIMD.float32x4.max(SIMD.float32x4(V.w[x][y][d], V.w[x][y][d+1], V.w[x][y][d+2], V.w[x][y][d+3]), zeroes);
          V2.w[x][y][d] = vv2.x;
          V2.w[x][y][d+1] = vv2.y;
          V2.w[x][y][d+2] = vv2.z;
          V2.w[x][y][d+3] = vv2.w;
        }
      }
    }

    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    let V = this.in_act; // we need to set dw of this
    let V2 = this.out_act;

    let {sx, sy, depth} = V;

    let zeroes = SIMD.float32x4.splat(0.0);

    for(let x = 0; x < sx; x++){
      for(let y = 0; y < sy; y++){
        for(let d = 0; d < depth; d += 4){
          let vv2 = SIMD.float32x4.max(SIMD.float32x4(V.dw[x][y][d], V.dw[x][y][d+1], V.dw[x][y][d+2], V.dw[x][y][d+3]), zeroes);
          V2.dw[x][y][d] = vv2.x;
          V2.dw[x][y][d+1] = vv2.y;
          V2.dw[x][y][d+2] = vv2.z;
          V2.dw[x][y][d+3] = vv2.w;
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