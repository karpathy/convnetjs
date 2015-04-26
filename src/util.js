import * as VolType from "./structures/vol.js";

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

export function randn(mu, std){ 
  return mu+gaussRandom()*std; 
}

// Array utilities

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
export function weightedSample(lst, probs) {
  var p = randf(0, 1.0);
  var cumprob = 0.0;
  for(var k=0 ,n=lst.length;k<n;k++) {
    cumprob += probs[k];
    if(p < cumprob) { return lst[k]; }
  }
}

export function augment(V, crop, dx = randi(0, V.sx - crop), dy = randi(0, V.sy - crop), fliplr = false) {
  
  // randomly sample a crop in the input volume
  var W;
  if(crop !== V.sx || dx!==0 || dy!==0) {
    W = new (new VolType(crop, crop, V.depth))();
    for(var x=0;x<crop;x++) {
      for(var y=0;y<crop;y++) {
        if(x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy) continue; // oob
        for(var d=0;d<V.depth;d++) {
         W[x][y][d] = V[x+dx][y+dy][d]; // copy data over
        }
      }
    }
  } else {
    W = V;
  }

  if(fliplr) {
    // flip volume horziontally
    var W2 = new W.constructor();
    for(var x=0;x<W.sx;x++) {
      for(var y=0;y<W.sy;y++) {
        for(var d=0;d<W.depth;d++) {
         W2[x][y][d] = W[W.sx - x - 1][y][d]; // copy data over
        }
      }
    }
    W = W2; //swap
  }
  return W;
}

// returns min, max and indices of an array

export function maxmin(w) {
  if(w.length === 0) { return {}; } // ... ;s

  var maxv = w[0];
  var minv = w[0];
  var maxi = 0;
  var mini = 0;
  for(var i=1;i<w.length;i++) {
    if(w[i] > maxv) { 
      maxv = w[i]; 
      maxi = i; 
    } 
    if(w[i] < minv) { 
      minv = w[i]; mini = i; 
    } 
  }
  return {
    maxi: maxi, 
    maxv: maxv, 
    mini: mini, 
    minv: minv, 
    dv:maxv-minv
  };
}