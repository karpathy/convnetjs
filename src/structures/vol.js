// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t. 
// the data. c is optionally a value to initialize the volume
// with. If c is missing, fills the Vol with random numbers.

export default function VolType(sx = 1, sy = 1, depth = 1){
  let VolType = new TypedObject.StructType({
    w : TypedObject.float64.array(depth).array(sy).array(sx),
    dw : TypedObject.float64.array(depth).array(sy).array(sx)
  });

  VolType.prototype.sx = sx;
  VolType.prototype.sy = sy;
  VolType.prototype.depth = depth;
  Voltype.prototype.toJSON = function toJSON(){
  	return {
  		sx : this.sx,
  		sy : this.sy,
  		depth : this.depth,
  		w : this.w,
  		dw : this.dw
  	}
  };

  return VolType;
}

export function fromJSON(json){
	if(typeof json === 'string'){
		json = JSON.parse(json);
	}
	return new (new VolType(json.sx, json.sy, json.depth))({w:json.w, dw:json.dw});
}