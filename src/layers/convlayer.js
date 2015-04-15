import Vol from "./convnet_vol.js";

export default class ConvLayer {

  constructor(opt = {}){

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 1; // stride at which we apply filters to input volume
    this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    // initializations
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.filters = [];
    for(var i=0;i<this.out_depth;i++) { 
      this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); 
    }
    this.biases = new Vol(1, 1, this.out_depth, bias);

  }

  forward(V, is_training) {
    // optimized code by @mdda that achieves 2x speedup over previous version

    this.in_act = V;
    var A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
    
    var V_sx = V.sx |0;
    var V_sy = V.sy |0;
    var xy_stride = this.stride |0;

    for(var d=0;d<this.out_depth;d++) {
      var f = this.filters[d];
      var x = -this.pad |0;
      var y = -this.pad |0;
      for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
        x = -this.pad |0;
        for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

          // convolve centered at this particular location
          var a = 0.0;
          for(var fy=0;fy<f.sy;fy++) {
            var oy = y+fy; // coordinates in the original input array coordinates
            for(var fx=0;fx<f.sx;fx++) {
              var ox = x+fx;
              if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                for(var fd=0;fd<f.depth;fd++) {
                  // avoid function call overhead (x2) for efficiency, compromise modularity :(
                  a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
                }
              }
            }
          }
          a += this.biases.w[d];
          A.set(ax, ay, d, a);
        }
      }
    }
    this.out_act = A;
    return this.out_act;
  }

  backward() {

    var V = this.in_act;
    V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

    var V_sx = V.sx |0;
    var V_sy = V.sy |0;
    var xy_stride = this.stride |0;

    for(var d=0;d<this.out_depth;d++) {
      var f = this.filters[d];
      var x = -this.pad |0;
      var y = -this.pad |0;
      for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
        x = -this.pad |0;
        for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

          // convolve centered at this particular location
          var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
          for(var fy=0;fy<f.sy;fy++) {
            var oy = y+fy; // coordinates in the original input array coordinates
            for(var fx=0;fx<f.sx;fx++) {
              var ox = x+fx;
              if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                for(var fd=0;fd<f.depth;fd++) {
                  // avoid function call overhead (x2) for efficiency, compromise modularity :(
                  var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                  var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                  f.dw[ix2] += V.w[ix1]*chain_grad;
                  V.dw[ix1] += f.w[ix2]*chain_grad;
                }
              }
            }
          }
          this.biases.dw[d] += chain_grad;
        }
      }
    }
  }

  getParamsAndGrads() {
    var response = [];
    for(var i=0;i<this.out_depth;i++) {
      response.push({params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul});
    }
    response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
    return response;
  }

  toJSON() {
    var json = {};
    json.sx = this.sx; // filter size in x, y dims
    json.sy = this.sy;
    json.stride = this.stride;
    json.in_depth = this.in_depth;
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.l1_decay_mul = this.l1_decay_mul;
    json.l2_decay_mul = this.l2_decay_mul;
    json.pad = this.pad;
    json.filters = this.filters.mapPar(x => x.toJSON());
    json.biases = this.biases.toJSON();
    return json;
  }

}

export function fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.sx = json.sx; // filter size in x, y dims
    this.sy = json.sy;
    this.stride = json.stride;
    this.in_depth = json.in_depth; // depth of input volume
    this.filters = [];
    this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
    this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
    this.pad = typeof json.pad !== 'undefined' ? json.pad : 0;
    this.filters = json.filters.mapPar(x => {
      let v = new Vol(0,0,0,0);
      v.fromJSON(x);
      return v;
    });
    this.biases = new Vol(0,0,0,0);
    this.biases.fromJSON(json.biases);
  }