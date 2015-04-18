import Vol from "./convnet_vol.js";

// Random number utilities
var return_v = false;
var v_val = 0.0;

export function gaussRandom() {
  if(return_v) { 
    return_v = false;
    return v_val; 
  }
  const [u, v] = [2*Math.random()-1, 2*Math.random()-1];
  var r = u*u + v*v;
  if(r == 0 || r > 1) {
    return gaussRandom();
  } else {
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
}

export function randf(a, b) { 
  return Math.random()*(b-a)+a; 
}

export function randi(a, b) { 
  return Math.floor(Math.random()*(b-a)+a); 
}

export function randn(mu, std){ 
  return mu+gaussRandom()*std; 
}

// Array utilities
export function zeros(n = 0) {
  if(typeof ArrayBuffer === 'undefined') {
    // lacking browser support
    var arr = new Array(n);
    for(var i=0;i<n;i++) { 
      arr[i]= 0; 
    }
    return arr;
  } else {
    return new Float64Array(n);
  }
}

export function arrUnique(arr) {
  return arr.filter((x, i) => {return (arr.indexOf(x) >= i)});
}

// return max and min of a given non-empty array.
export function maxmin(w) {
  if(w.length === 0) { return {}; } // ... ;s
  var maxv = w[0];
  var minv = w[0];
  var maxi = 0;
  var mini = 0;
  var n = w.length;
  for(var i=1;i<n;i++) {
    if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
    if(w[i] < minv) { minv = w[i]; mini = i; } 
  }
  return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
}

// create random permutation of numbers, in range [0...n-1]
export function randperm(n) {
  var i = n,
      j = 0,
      temp;
  var array = [];
  for(var q=0;q<n;q++)array[q]=q;
  while (i--) {
      j = Math.floor(Math.random() * (i+1));
      temp = array[i];
      array[i] = array[j];
      array[j] = temp;
  }
  return array;
}

// sample from list lst according to probabilities in list probs
// the two lists are of same size, and probs adds up to 1
export weightedSample(lst, probs) {
  var p = randf(0, 1.0);
  var cumprob = 0.0;
  for(var k=0,n=lst.length;k<n;k++) {
    cumprob += probs[k];
    if(p < cumprob) { return lst[k]; }
  }
}

// syntactic sugar function for getting default parameter values
export function getopt(opt, field_name, default_value) {
  if(typeof field_name === 'string') {
    // case of single string
    return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
  } else {
    // assume we are given a list of string instead
    var ret = default_value;
    for(var i=0;i<field_name.length;i++) {
      var f = field_name[i];
      if (typeof opt[f] !== 'undefined') {
        ret = opt[f]; // overwrite return value
      }
    }
    return ret;
  }
}

export function assert(condition, message) {
  if (!condition) {
    message = message || "Assertion failed";
    if (typeof Error !== "undefined") {
      throw new Error(message);
    }
    throw message; // Fallback
  }
}

export function augment(V, crop, dx = randi(0, V.sx - crop), dy = randi(0, V.sy - crop), fliplr = false) {
  
  // randomly sample a crop in the input volume
  var W;
  if(crop !== V.sx || dx!==0 || dy!==0) {
    W = new Vol(crop, crop, V.depth, 0.0);
    for(var x=0;x<crop;x++) {
      for(var y=0;y<crop;y++) {
        if(x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy) continue; // oob
        for(var d=0;d<V.depth;d++) {
         W.set(x,y,d,V.get(x+dx,y+dy,d)); // copy data over
        }
      }
    }
  } else {
    W = V;
  }

  if(fliplr) {
    // flip volume horziontally
    var W2 = W.cloneAndZero();
    for(var x=0;x<W.sx;x++) {
      for(var y=0;y<W.sy;y++) {
        for(var d=0;d<W.depth;d++) {
         W2.set(x,y,d,W.get(W.sx - x - 1,y,d)); // copy data over
        }
      }
    }
    W = W2; //swap
  }
  return W;
}

export function imageDataToVol(imgdata, convert_grayscale = false){

  var ImageDataVol = new VolType(imgdata.width, imgdata.height, 4);

  // prepare the input: get pixels and normalize them
  var p = img_data.data;
  var W = img_data.width;
  var H = img_data.height;
  var pv = new Float64Array(img_data.data.length);

  const tff = SIMD.float32x4.splat(255.0);
  const mpf = SIMD.float32x4.splat(0.5);
  const len = p.length / 4;

  // normalize image pixels to [-0.5, 0.5]
  for(var i=0; i < len; i++) {
    let res = SIMD.float32x4.sub(SIMD.float32x4.div(SIMD.float32x4(p[i], p[i+1], p[i+2], p[i+3]), tff), mpf);
    pv[i*4] = res.x;
    pv[i*4+1] = res.y;
    pv[i*4+2] = res.z;
    pv[i*4+3] = res.w;
  }

  var x = new Vol(W, H, 4, 0.0); //input volume (image)
  x.w = pv;

  if(convert_grayscale) {
    // flatten into depth=1 array
    var x1 = new Vol(W, H, 1, 0.0);
    for(var i=0;i<W;i++) {
      for(var j=0;j<H;j++) {
        x1.set(i,j,0,x.get(i,j,0));
      }
    }
    x = x1;
  }

  return x;
}

export function imageToVol(img, convert_grayscale = false) {

  var canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  var ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  return imageDataToVol(ctx.getImageData(0, 0, canvas.width, canvas.height), convert_grayscale);
}

// a window stores _size_ number of values
// and returns averages. Useful for keeping running
// track of validation or training accuracy during SGD

export class Window {

  constructor(size = 100, minsize = 20){
    this.v = [];
    this.size = typeof(size)==='undefined' ? 100 : size;
    this.minsize = typeof(minsize)==='undefined' ? 20 : minsize;
    this.sum = 0;
  }

  add(x) {
    this.v.push(x);
    this.sum += x;
    if(this.v.length>this.size) {
      var xold = this.v.shift();
      this.sum -= xold;
    }
  }

  get_average() {
    if(this.v.length < this.minsize) return -1;
    else return this.sum/this.v.length;
  }

  reset(x) {
    this.v = [];
    this.sum = 0;
  }

}

// returns min, max and indices of an array

export function maxmin(w) {
  if(w.length === 0) { return {}; } // ... ;s

  var maxv = w[0];
  var minv = w[0];
  var maxi = 0;
  var mini = 0;
  for(var i=1;i<w.length;i++) {
    if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
    if(w[i] < minv) { minv = w[i]; mini = i; } 
  }
  return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
}

// returns string representation of float
// but truncated to length of d digits

export function f2t(x, d) {
  if(typeof(d)==='undefined') { var d = 5; }
  var dd = 1.0 * Math.pow(10, d);
  return '' + Math.floor(x*dd)/dd;
}