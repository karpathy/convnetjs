import Vol from "./convnet_vol.js";

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size

export class MaxoutLayer {

  constructor(opt = {}){
    // required
    this.group_size = typeof opt.group_size !== 'undefined' ? opt.group_size : 2;

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = Math.floor(opt.in_depth / this.group_size);
    this.layer_type = 'maxout';

    this.switches = global.zeros(this.out_sx*this.out_sy*this.out_depth); // useful for backprop
  }

  forward(V, is_training) {
    this.in_act = V;
    var N = this.out_depth; 
    var V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

    // optimization branch. If we're operating on 1D arrays we dont have
    // to worry about keeping track of x,y,d coordinates inside
    // input volumes. In convnets we do :(
    if(this.out_sx === 1 && this.out_sy === 1) {
      for(var i=0;i<N;i++) {
        var ix = i * this.group_size; // base index offset
        var a = V.w[ix];
        var ai = 0;
        for(var j=1;j<this.group_size;j++) {
          var a2 = V.w[ix+j];
          if(a2 > a) {
            a = a2;
            ai = j;
          }
        }
        V2.w[i] = a;
        this.switches[i] = ix + ai;
      }
    } else {
      var n=0; // counter for switches
      for(var x=0;x<V.sx;x++) {
        for(var y=0;y<V.sy;y++) {
          for(var i=0;i<N;i++) {
            var ix = i * this.group_size;
            var a = V.get(x, y, ix);
            var ai = 0;
            for(var j=1;j<this.group_size;j++) {
              var a2 = V.get(x, y, ix+j);
              if(a2 > a) {
                a = a2;
                ai = j;
              }
            }
            V2.set(x,y,i,a);
            this.switches[n] = ix + ai;
            n++;
          }
        }
      }

    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    var V = this.in_act; // we need to set dw of this
    var V2 = this.out_act;
    var N = this.out_depth;
    V.dw = global.zeros(V.w.length); // zero out gradient wrt data

    // pass the gradient through the appropriate switch
    if(this.out_sx === 1 && this.out_sy === 1) {
      for(var i=0;i<N;i++) {
        var chain_grad = V2.dw[i];
        V.dw[this.switches[i]] = chain_grad;
      }
    } else {
      // bleh okay, lets do this the hard way
      var n=0; // counter for switches
      for(var x=0;x<V2.sx;x++) {
        for(var y=0;y<V2.sy;y++) {
          for(var i=0;i<N;i++) {
            var chain_grad = V2.get_grad(x,y,i);
            V.set_grad(x,y,this.switches[n],chain_grad);
            n++;
          }
        }
      }
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    var json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.group_size = this.group_size;
    return json;
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
    this.group_size = json.group_size;
    this.switches = global.zeros(this.group_size);
  }

}