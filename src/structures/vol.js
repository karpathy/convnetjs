// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t. 
// the data. 

export default function VolType(sx = 1, sy = 1, depth = 1){
  let VolType = new TypedObject.StructType({
    w : TypedObject.float64.array(depth).array(sy).array(sx),
    dw : TypedObject.float64.array(depth).array(sy).array(sx)
  });

  Object.defineProperty(VolType.prototype, 'sx', {
    enumerable : true,
    value : sx,
    writable : false
  });

  Object.defineProperty(VolType.prototype, 'sy', {
    enumerable : true,
    value : sy,
    writable : false
  });

  Object.defineProperty(VolType.prototype, 'depth', {
    enumerable : true,
    value : depth,
    writable : false
  });

  Voltype.prototype.toJSON = function toJSON(){
  	return this;
  };

  return VolType;
}

/*
 * Rather than make a new VolType, these two functions actually return a new Vol. 
 * The VolType can be found using the constructor property of the Vols.
 */

export function fromJSON(json){
	if(typeof json === 'string'){
		json = JSON.parse(json);
	}
	return new (new VolType(json.sx, json.sy, json.depth))({w:json.w, dw:json.dw});
}

export function fromImage(image, convert_grayscale = false){
  let canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  let ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  return imageDataToVol(ctx.getImageData(0, 0, canvas.width, canvas.height), convert_grayscale);
}

export function fromImageData(imagedata, convert_grayscale = false){

  let x;

  // prepare the input: get pixels and normalize them
  let v = new (new VolType(imgdata.width, imgdata.height, 4))({w:(new Float32Array(imagedata.data)).buffer});
  let vd = new Float32Array(TypedObject.storage(v.w).buffer);

  const tff = SIMD.float32x4.splat(255.0);
  const mpf = SIMD.float32x4.splat(0.5);
  const len = p.length;

  // normalize image pixels to [-0.5, 0.5]
  for(let i = 0; i < len; i += 4){
    SIMD.float32x4.store(vd, i, SIMD.float32x4.sub(SIMD.float32x4.div(SIMD.float32x4.load(vd, i), tff), mpf))
  }

  if(convert_grayscale) {

    let g = new (new VolType(imgdata.width, imgdata.height, 1))(); //input volume (image)

    let len = imgdata.width * imgdata.height;

    // flatten into depth=1 array
    
    for(let i = 0, j = 0; i < len; i += 4, j += 16){
      SIMD.float32x4.store(g, i, SIMD.float32x4.div(SIMD.float32x4.add(SIMD.float32x4.add(SIMD.float32x4(vd, vd+4, vd+8, vd+12), SIMD.float32x4(vd+1, vd+5, vd+9, vd+13)), SIMD.float32x4(vd+2, vd+6, vd+10, vd+14)), SIMD.float32x4.splat(3)))
    }

    x = g;
  }else{
    x = v;
  }

  return x;
}