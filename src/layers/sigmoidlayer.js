import * as Layer from "./layer.js";

// Implements Sigmoid nonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.

export class SigmoidLayer extends Layer {

  constructor(opt = {}){
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'sigmoid';
  }

  forward(V, is_training) {
    this.out_act = new V.constructor();
    let {sx, sy, depth} = V2;
    let ones = SIMD.float32x4.splat(0.0);
    for(let x = 0; x < sx; x++){
      for(let y = 0; y < sy; y++){
        for(let d = 0; d < depth; d += 4){
          let vv2 = SIMD.float32x4.div(ones, SIMD.float32x4.add(SIMD.float32x4(Math.exp(-this.in_act.w[x][y][d]), Math.exp(-this.in_act.w[x][y][d+1]), Math.exp(-this.in_act.w[x][y][d+2]), Math.exp(-this.in_act.w[x][y][d+3]), zeroes));
          this.out_act.w[x][y][d] = vv2.x; this.out_act.w[x][y][d+1] = vv2.y; this.out_act.w[x][y][d+2] = vv2.z; this.out_act.w[x][y][d+3] = vv2.w;
        }
      }
    }
    return this.out_act;
  }

  backward() {
    let [sx, sy, depth] = [this.in_act.sx, this.in_act.sy, this.in_act.depth];
    for(let x = 0; x < sx; x++){
      for(let y = 0; y < sy; y++){
        for(let d = 0; y < depth; d++){
          let dep = SIMD.float32x4(this.out_act.w[x][y][d], this.out_act.w[x][y][d+1], this.out_act.w[x][y][d+2], this.out_act.w[x][y][d+3]);
          let res = SIMD.float32x4.mul(SIMD.float32x4.mul(dep, SIMD.float32x4.sub(SIMD.float32x4.splat(1.0), dep)), SIMD.float32x4(this.out_act.dw[x][y][d], this.out_act.dw[x][y][d+1], this.out_act.dw[x][y][d+2], this.out_act.dw[x][y][d+3]););
          this.in_act.dw[x][y][d] = res.x; this.in_act.dw[x][y][d+1] = res.y; this.in_act.dw[x][y][d+2] = res.z; this.in_act.dw[x][y][d+3] = res.w;
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
    };
  }

}

export function fromJSON(json) {
  if(typeof json == 'string'){
    json = JSON.parse(json);
  }
  return new SigmoidLayer({
    out_depth : json.out_depth,
    out_sx : json.out_sx,
    out_sy : json.out_sy
  });
}