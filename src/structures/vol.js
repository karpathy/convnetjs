// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t. 
// the data. 

export default class VolType extends StructType {
  constructor(sx = 1, sy = 1, depth = 1){
    super({
      w : new ArrayType(new ArrayType(new ArrayType(float64, depth), sy), sx),
      dw : new ArrayType(new ArrayType(new ArrayType(float64, depth), sy), sx)
    });
  }
  
  get sx(){
    return this.w.length;
  }
  
  get sy(){
    return this.w[0].length;
  }
  
  get depth(){
    return this.w[0][0].length;
  }

  static fromJSON(json){
    if(typeof json === 'string'){
      json = JSON.parse(json);
    }
    return new (new VolType(json.sx, json.sy, json.depth))({w:json.w, dw:json.dw});
  }

  static fromImage(image, convert_grayscale = false){
    let canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    return imageDataToVol(ctx.getImageData(0, 0, canvas.width, canvas.height), convert_grayscale);
  }

  static fromImageData(imagedata, convert_grayscale = false){

    let x;

    // prepare the input: get pixels and normalize them
    let v = new (new VolType(imgdata.width, imgdata.height, 4))({w:(new Float64Array(imagedata.data)).buffer});
    let vd = new Float64Array(storage(v.w).buffer);

    const tff = SIMD.float64x2.splat(255.0);
    const mpf = SIMD.float64x2.splat(0.5);
    const len = p.length;

    // normalize image pixels to [-0.5, 0.5]
    for(let i = 0; i < len; i += 2){
      SIMD.float64x2.store(vd, i, SIMD.float64x2.sub(SIMD.float64x2.div(SIMD.float64x2.load(vd, i), tff), mpf))
    }

    if(convert_grayscale) {

      let g = new (new VolType(imgdata.width, imgdata.height, 1))(); //input volume (image)

      let len = imgdata.width * imgdata.height;

      // flatten into depth=1 array
      for(let i = 0, j = 0; i < len; i += 2, j += 8){
        SIMD.float64x2.store(g, i, SIMD.float64x2.div(SIMD.float64x2.add(SIMD.float64x2.add(SIMD.float64x2(vd[i], vd[i+4]), SIMD.float64x2(vd[i+1], vd[i+5])), SIMD.float64x2(vd[i+2], vd[i+6])), SIMD.float64x2.splat(3)))
      }

      x = g;
    }else{
      x = v;
    }

    return x;
  }
}