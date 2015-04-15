// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t. 
// the data. c is optionally a value to initialize the volume
// with. If c is missing, fills the Vol with random numbers.

export default class Vol {

  constructor(sx = 1, sy = 1, depth = 1, c){
    // This is a nice way to check if it's an array
    if(sx.constructor.name.indexOf('Array') > -1) {
      // we were given a list in sx, assume 1D volume and fill it up
      this.sx = 1;
      this.depth = sx.length;

      // we have to do the following copy because we want to use
      // fast typed arrays, not an ordinary javascript array
      this.dw = zeros(this.depth);
      // Passing an array of numbers to a TypedArray constructor 
      // will fill the array quicker than a for loop.
      this.w = new Float64Array(sx); 
    } else {
      // we were given dimensions of the vol
      this.sx = sx;
      this.depth = depth;
      var n = sx*sy*depth;
      this.w = zeros(n);
      this.dw = zeros(n);
      if(typeof c === 'undefined') {
        // weight normalization is done to equalize the output
        // variance of every neuron, otherwise neurons with a lot
        // of incoming connections have outputs of larger variance
        var scale = Math.sqrt(1.0/(sx*sy*depth));
        for(var i=0;i<n;i++) { 
          this.w[i] = randn(0.0, scale);
        }
      } else {
        for(var i=0;i<n;i++) { 
          this.w[i] = c;
        }
      }
    }
    this.sy = sy;
  }

  get(x, y, d) { 
    return this.w[((this.sx * y)+x)*this.depth+d];
  }

  set(x, y, d, v) { 
    this.w[((this.sx * y)+x)*this.depth+d] = v; 
  }

  add(x, y, d, v) { 
    this.w[((this.sx * y)+x)*this.depth+d] += v; 
  }

  getGrad(x, y, d) { 
    return this.dw[((this.sx * y)+x)*this.depth+d]; 
  }

  setGrad(x, y, d, v) { 
    this.dw[((this.sx * y)+x)*this.depth+d] = v; 
  }

  addGrad(x, y, d, v) { 
    this.dw[((this.sx * y)+x)*this.depth+d] += v; 
  }

  cloneAndZero() { 
    return new Vol(this.sx, this.sy, this.depth, 0.0);
  }

  clone() {
    var V = new Vol(this.sx, this.sy, this.depth, 0);
    V.w = this.w.mapPar(x => x);
    return V;
  }

  addFrom(V) { 
    for(var k=0;k<this.w.length;k++) { 
      this.w[k] += V.w[k]; 
    }
  }

  addFromScaled(V, a) { 
    for(var k=0;k<this.w.length;k++) { 
      this.w[k] += a*V.w[k]; 
    }
  }

  setConst(a) { 
    this.w = (new Array(this.w.length)).map(x => a);
  }

  toJSON() {
    // todo: we may want to only save d most significant digits to save space
    return {
      sx : this.sx,
      sy : this.sy,
      depth : this.depth,
      w : this.w
    };
    // we wont back up gradients to save space
  }

  fromJSON(json) {
    this.sx = json.sx;
    this.sy = json.sy;
    this.depth = json.depth;
    var n = this.sx*this.sy*this.depth;
    this.dw = zeros(n);
    this.w = json.w.map(x => x);
  }
}