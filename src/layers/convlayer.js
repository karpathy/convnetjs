import * as VolType from "../structures/vol.js";
import * as Layer from "./layer.js";

export default class ConvLayer extends Layer{

  constructor(opt = {}){
    super(opt);

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = opt.sy || this.sx;
    this.stride = opt.stride || 1; // stride at which we apply filters to input volume
    this.pad = opt.pad || 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = opt.l1_decay_mul || 0.0;
    this.l2_decay_mul = opt.l2_decay_mul || 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    this.out_type = new VolType(this.out_sx, this.out_sy, this.out_depth);

    // initializations
    let bias = opt.bias_pref || 0.0;
    this.bias_type = new VolType(1, 1, this.out_depth);
    this.biases = new this.bias_type({w:[[(new Float64Array()).map(x => bias)]]});

    this.filter_type = new VolType(this.sx, this.sy, this.in_depth);
    this.filters = new (this.filter_type.array(this.out_depth))();
    for(var i = 0; i < this.out_depth; i++) { 
      this.filters[i] = new this.filter_type(); 
    }

  }

  forward(V, is_training = false) {
    // optimized code by @mdda that achieves 2x speedup over previous version

    this.in_act = V;
    this.out_act = new this.out_type();

    let vw = new Float64Array(TypedObject.storage(this.in_act.w).buffer);
    let vd = new Float64Array(TypedObject.storage(this.in_act.dw).buffer);
    let v2w = new Float64Array(TypedObject.storage(this.out_act.w).buffer);
    let v2d = new Float64Array(TypedObject.storage(this.out_act.dw).buffer);
    let b = new Float64Array(TypedObject.storage(this.biases.w).buffer);

    let x = (0|0), 
        y = (0|0);
    let chain_grad = +0.0,
        oy = +0.0,
        ox = +0.0;
    let a = SIMD.float64x2.zero();

    for(let d = 0 ; d < this.out_depth; d++) {
      let {fsx, fsy, fdep} = this.filters[d];
      let fw = new Float64Array(TypedObject.storage(this.filters[d].w).buffer);
      x = -this.pad | 0;
      y = -this.pad | 0;
      for(let ay = 0; ay < this.out_sy; y += this.stride, ay++) {  // xy_stride
        x = -this.pad | 0;
        for(let ax = 0; ax < this.out_sx; x += this.stride, ax++) {  // xy_stride
          // convolve centered at this particular location
          a = SIMD.float64x2.zero();
          for(let fy = 0; fy < fsy; fy++) {
            oy = y + fy; // coordinates in the original input array coordinates
            for(let fx = 0; fx < fsx; fx++) {
              ox = x + fx;
              if(oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                let xco = ((v.sx*oy)+ox)*v.depth,
                    fco = ((fsx*fy)+fx)*fdep;
                for(let fd = 0; fd < fdep; fd++) {
                  // avoid function call overhead (x2) for efficiency, compromise modularity :(
                  a = SIMD.float64x2.add(a, SIMD.float64x2.mul(SIMD.float64x2.mul(SIMD.float64x2.load(fw, fco+fd), SIMD.float64x2.load(vw, xco+fd)),SIMD.float64x2(fd<fdep?1:0,fd+1<fdep?1:0)));
                }
              }
            }
          }
          v2w[ax][ay][d] = a.x + a.y + b[d];
        }
      }
    }
    return this.out_act;
  }

  backward() {

    let vw = new Float64Array(TypedObject.storage(this.in_act.w).buffer);
    let vd = new Float64Array(TypedObject.storage(this.in_act.dw).buffer);
    let v2w = new Float64Array(TypedObject.storage(this.out_act.w).buffer);
    let v2d = new Float64Array(TypedObject.storage(this.out_act.dw).buffer);
    let b = new Float64Array(TypedObject.storage(this.biases.dw).buffer);

    let x = (0|0),
        y = (0|0),
        s = (this.stride|0),
        ;
    let oy = +0.0,
        ox = +0.0;

    for(let i = 0; i < vd.length; i++){
      vd[i] = 0;
    }

    for(let d = 0; d < this.out_depth; d++){
      let {sx, sy, depth} = this.filters[d];
      let fd = new Float64Array(TypedObject.storage(this.filters[d].dw).buffer;
      let fw = new Float64Array(TypedObject.storage(this.filters[d].w).buffer);
      x = -this.pad | 0;
      y = -this.pad | 0;
      for(let ay = 0; ay < this.out_sy; y += s, ay++) {
        x = -this.pad | 0;
        for(let ax = 0; ax < this.out_sx; x += s, ax++) {
          let chain_grad = SIMD.float64x2.splat(v2d[ax][ay][d]);
          for(let fy = 0; fy < sy; fy++) {
            oy = +(y + fy);
            for(let fx = 0; fx < sx; fx++){
              ox = +(x + fx);
              if(oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                let xco = ((v.sx*oy)+ox)*v.depth,
                    fco = ((fsx*fy)+fx)*fdep,
                    qual = SIMD.float64x2.splat(1.0);
                for(let fd = 0; fd < depth; fd += 2){
                  qual = SIMD.float64x2(fd < depth ? 1 : 0, fd + 1 < depth ? 1 : 0);
                  SIMD.float64x2.store(fd, fco+fd, SIMD.float64x2.add(SIMD.float64x2.load(fd, fco+fd), SIMD.float64x2.mul(qual, SIMD.float64x2.mul(chain_grad, SIMD.float64x2.load(vw, xco+fd)))));
                  SIMD.float64x2.store(vd, fco+fd, SIMD.float64x2.add(SIMD.float64x2.load(vd, fco+fd), SIMD.float64x2.mul(qual, SIMD.float64x2.mul(chain_grad, SIMD.float64x2.load(fw, fco+fd)))));
                }
              }
            }
          }
          b[d] += v2d[ax][ay][d];
        }
      }
    }
  }

  getParamsAndGrads() {
    var response = new Array(this.out_depth + 1);
    for(var i=0;i<this.out_depth;i++) {
      response.push({
        params: this.filters[i].w, 
        grads: this.filters[i].dw, 
        l2_decay_mul: this.l2_decay_mul, 
        l1_decay_mul: this.l1_decay_mul
      });
    }
    response.push({
      params: this.biases.w, 
      grads: this.biases.dw, 
      l1_decay_mul: 0.0, 
      l2_decay_mul: 0.0
    });
    return response;
  }

  toJSON() {
    return {
      sx : this.sx, // filter size in x, y dims
      sy : this.sy,
      stride : this.stride,
      in_depth : this.in_depth,
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      l1_decay_mul : this.l1_decay_mul,
      l2_decay_mul : this.l2_decay_mul,
      pad : this.pad,
      filters : this.filters.mapPar(x => x.toJSON()),
      biases : this.biases.toJSON()
    };
  }

}

export function fromJSON(json) {
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new ConvLayer({
    out_depth : json.out_depth,
    out_sx : json.out_sx,
    out_sy : json.out_sy,
    layer_type : json.layer_type,
    sx : json.sx, // filter size in x, y dims
    sy : json.sy,
    stride : json.stride,
    in_depth : json.in_depth, // depth of input volume
    l1_decay_mul : json.l1_decay_mul,
    l2_decay_mul : json.l2_decay_mul,
    pad : json.pad,
    filters : json.filters.mapPar(x => VolType.fromJSON(x)),
    biases : VolType.fromJSON(json.biases)
  });
}