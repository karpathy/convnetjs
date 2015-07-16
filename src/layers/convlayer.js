import VolType from "../structures/vol.js";
import Layer from "./layer.js";

export default class ConvLayer extends Layer {

  constructor(options){
    super(options);

    // required
    this.out_depth = options.filters;
    this.sx = options.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = options.in_depth;
    this.in_sx = options.in_sx;
    this.in_sy = options.in_sy;
    
    // optional
    this.sy = options.sy || this.sx;
    this.stride = options.stride || 1; // stride at which we apply filters to input volume
    this.pad = options.pad || 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = options.l1_decay_mul || 0.0;
    this.l2_decay_mul = options.l2_decay_mul || 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    this.out_type = new VolType(this.out_sx, this.out_sy, this.out_depth);

    // initializations
    this.bias_type = new VolType(1, 1, this.out_depth);
    this.biases = new this.bias_type({w:[[(new Float64Array(this.out_depth)).map(x => (opt.bias_pref || 0.0))]]});

    this.filter_type = new VolType(this.sx, this.sy, this.in_depth);
    this.filters = new (this.filter_type.array(this.out_depth))([for (filter of (function* (n){
      for(var i = 0; i < this.out_depth; i++) { 
       yield new this.filter_type(); 
      }
    })(this.out_depth)) filter]);

  }

  forward(V, is_training = false) {

    super.forward(V, use_webgl, is_training);
    // optimized code by @mdda that achieves 2x speedup over previous version
    this.out_act = new this.out_type();

    let vw = new Float64Array(storage(this.in_act.w).buffer);
    let vd = new Float64Array(storage(this.in_act.dw).buffer);
    let v2w = new Float64Array(storage(this.out_act.w).buffer);
    let v2d = new Float64Array(storage(this.out_act.dw).buffer);
    let b = new Float64Array(storage(this.biases.w).buffer);
    let fw = new Float64Array(storage(this.filters).buffer);
    let x = (0|0), y = (0|0),
        xco = (0|0), fco = (0|0),
        fsx = (this.sx|0), fsy = (this.sy|0), fdep = (this.in_depth|0), fstep = (this.filters.byteLength/8/this.filters.length|0),
        fd = (0|0), fx = (0|0), fy = (0|0), ax = (0|0), ay = (0|0),
        oy = (0|0), ox = (0|0),
        pad = (-this.pad|0), stride = (this.stride|0),
        osy = (this.out_sy|0), osx = (this.out_sx|0), odep = (this.out_depth|0),
        isx = (this.in_sx|0), isy = (this.in_sy|0);
    let a = SIMD.float64x2.zero();

    for(let d = 0 ; d < odep; d++) {
      x = pad | 0;
      y = pad | 0;
      for(ay = 0; ay < osy; y += stride, ay++) {  // xy_stride
        x = pad | 0;
        for(ax = 0; ax < osx; x += stride, ax++) {  // xy_stride
          // convolve centered at this particular location
          a = SIMD.float64x2.zero();
          for(fy = 0; fy < fsy; fy++) {
            oy = ((y + fy)|0); // coordinates in the original input array coordinates
            for(fx = 0; fx < fsx; fx++) {
              ox = ((x + fx)|0);
              if(oy >= 0 && oy < isy && ox >= 0 && ox < isx) {
                xco = ((((isx*oy)+ox)*fdep)|0);
                fco = ((((fsx*fy)+fx)*fdep+fstep*d)|0);
                for(fd = 0; fd < fdep; fd++) {
                  // avoid function call overhead (x2) for efficiency, compromise modularity :(
                  a = SIMD.float64x2.add(a, SIMD.float64x2.mul(SIMD.float64x2.mul(SIMD.float64x2.load(fw, fco+fd), SIMD.float64x2.load(vw, xco+fd)),SIMD.float64x2(fd<fdep?1:0,fd+1<fdep?1:0)));
                }
              }
            }
          }
          v2w[(((osx*ay)+ax)*odep)] = +(a.x + a.y + b[d]);
        }
      }
    }

    return this.out_act;
  }

  backward(use_webgl = false, is_training = false) {

    let vw = new Float64Array(storage(this.in_act.w).buffer);
    let vd = new Float64Array(storage(this.in_act.dw).buffer);
    let v2w = new Float64Array(storage(this.out_act.w).buffer);
    let v2d = new Float64Array(storage(this.out_act.dw).buffer);
    let b = new Float64Array(storage(this.biases.dw).buffer);
    let fw = new Float64Array(storage(this.filters).buffer);
    let x = (0|0), y = (0|0),
        xco = (0|0), fco = (0|0),
        fsx = (this.sx|0), fsy = (this.sy|0), fdep = (this.in_depth|0), fstep = (this.filters.byteLength/8/this.filters.length|0),
        fd = (0|0), fx = (0|0), fy = (0|0), ax = (0|0), ay = (0|0),
        oy = (0|0), ox = (0|0),
        pad = (-this.pad|0), stride = (this.stride|0),
        osy = (this.out_sy|0), osx = (this.out_sx|0), odep = (this.out_depth|0),
        isx = (this.in_sx|0), isy = (this.in_sy|0);
    let chain_grad = +0.0;
    let a = SIMD.float64x2.zero(), qual = SIMD.float64x2.splat(1.0);

    for(let i = 0; i < vd.length; i++){
      vd[i] = 0;
    }

    for(let d = 0 ; d < odep; d++) {
      x = pad | 0;
      y = pad | 0;
      for(ay = 0; ay < osy; y += stride, ay++) {  // xy_stride
        x = pad | 0;
        for(ax = 0; ax < osx; x += stride, ax++) {  // xy_stride
          chain_grad = SIMD.float64x2.splat(v2d[ax][ay][d]);
          for(fy = 0; fy < fsy; fy++) {
            oy = ((y + fy)|0); // coordinates in the original input array coordinates
            for(fx = 0; fx < fsx; fx++) {
              ox = ((x + fx)|0);
              if(oy >= 0 && oy < isy && ox >= 0 && ox < isx) {
                xco = ((((isx*oy)+ox)*fdep)|0),
                fco = ((((fsx*fy)+fx)*fdep+fstep*d)|0);
                for(let fd = 0; fd < depth; fd += 2){
                  qual = SIMD.float64x2(((fd < depth ? 1 : 0)|0), ((fd + 1 < depth ? 1 : 0)|0));
                  SIMD.float64x2.store(fd, fco+fd, 
                    SIMD.float64x2.add(
                      SIMD.float64x2.load(fd, fco+fd), 
                      SIMD.float64x2.mul(qual, 
                        SIMD.float64x2.mul(chain_grad, 
                          SIMD.float64x2.load(vw, xco+fd)
                        )
                      )
                    )
                  );
                  SIMD.float64x2.store(vd, fco+fd, 
                    SIMD.float64x2.add(SIMD.float64x2.load(vd, fco+fd), 
                      SIMD.float64x2.mul(qual, 
                        SIMD.float64x2.mul(chain_grad, 
                          SIMD.float64x2.load(fw, fco+fd)
                        )
                      )
                    )
                  );
                }
              }
            }
          }
          b[d] += v2d[(((osx*ay)+ax)*odep)];
        }
      }
    }
  }

  getParamsAndGrads() {
    return [for (param of (function* (n){
      for(let i = 0; i < n; i++){
        yield {
          params: this.filters[i].w, 
          grads: this.filters[i].dw, 
          l2_decay_mul: this.l2_decay_mul, 
          l1_decay_mul: this.l1_decay_mul
        }
      }
    })(this.out_depth)) param].concat([{
      params: this.biases.w, 
      grads: this.biases.dw, 
      l1_decay_mul: 0.0, 
      l2_decay_mul: 0.0
    }]);
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