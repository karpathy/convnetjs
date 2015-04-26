export default class Layer { 

	constructor(opts){

	}

	forward(V, is_training = false){

	}

	backward(){

	}

	toJSON(){

	}

	get layerType(){
		return this.constructor.name;
	}

	static fromJSON(json = {}){
		if(typeof json === 'string'){
			json = JSON.parse(json);
		}
		return new this.constructor(json);
	}

}