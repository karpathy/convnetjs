import * as VolType from "../structures/vol.js";
import * as Layer from "./layer.js";

export default class PoolLayer extends Layer {

  constructor(opt = {}){
    super(opt);
    // required
    this.sx = opt.sx; // filter size
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;

    // optional
    this.sy = opt.sy || this.sx;
    this.stride = opt.stride || 2;
    this.pad = opt.pad || 0; // amount of 0 padding to add around borders of input volume

    // computed
    this.out_depth = this.in_depth;
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'pool';
    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchx = new Float64Array(this.out_sx*this.out_sy*this.out_depth);
    this.switchy = new Float64Array(this.out_sx*this.out_sy*this.out_depth);

    this.out_vol = new VolType(this.out_sx, this.out_sy, this.out_depth);

  }

  forward(V, use_webgl = false, is_training = false) {
    super.forward(V, is_training);

    var A = new this.out_voltype();
    
    var n=0; // a counter for switches
    for(var d=0;d<this.out_depth;d++) {
      var x = -this.pad;
      var y = -this.pad;
      for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
        y = -this.pad;
        for(var ay=0; ay<this.out_sy; y+=this.stride, ay++, n++) {

          // convolve centered at this particular location
          var a = -99999; // hopefully small enough ;\
          var winx=-1,winy=-1;
          for(var fx=0;fx<this.sx;fx++) {
            for(var fy=0;fy<this.sy;fy++) {
              var oy = y+fy;
              var ox = x+fx;
              if(oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                var v = V.w[ox][oy][d];
                // perform max pooling and store pointers to where
                // the max came from. This will speed up backprop 
                // and can help make nice visualizations in future
                if(v > a) { 
                  a = v; 
                  winx = ox; 
                  winy = oy;
                }
              }
            }
          }
          this.switchx[n] = winx;
          this.switchy[n] = winy;
          A.w[ax][ay][d] = a;
        }
      }
    }
    this.out_act = A;
    return this.out_act;
  }

  backward() { 
    // pooling layers have no parameters, so simply compute 
    // gradient wrt data here
    var V = this.in_act;
    V.dw = new Float64Array(V.w.length); // zero out gradient wrt data
    var A = this.out_act; // computed in forward pass 

    var n = 0;
    for(var d=0;d<this.out_depth;d++) {
      var x = -this.pad;
      var y = -this.pad;
      for(var ax = 0; ax < this.out_sx; x += this.stride, ax++) {
        y = -this.pad;
        for(var ay = 0; ay<this.out_sy; y += this.stride, ay++, n++) {
          V.dw[this.switchx[n]][this.switchy[n]][d] = this.out_act.dw[ax][ay][d];
        }
      }
    }
  }

  toJSON() {
    return {
      sx : this.sx,
      sy : this.sy,
      stride : this.stride,
      in_depth : this.in_depth,
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      pad : this.pad
    };
  }
  
}