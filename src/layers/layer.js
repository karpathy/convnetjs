export default class Layer { 

	constructor(opts){

	}

	forward(V, is_training = false) {
	    this.in_act = V;
	    this.out_act = V;
	    return V; // identity function
	  }

	backward(){

	}

	getParamsAndGrads(){
		return [];
	}

	toJSON(){
		return {};
	}

	static fromJSON(json = {}){
		if(typeof json === 'string'){
			json = JSON.parse(json);
		}
		return new this.constructor(json);
	}

}