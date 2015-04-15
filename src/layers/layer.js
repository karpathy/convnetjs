export default class Layer { 

	constructor(opts){

	}

	toJSON(){

	}

}

export function fromJSON(json = {}){
	if(typeof json === 'string'){
		json = JSON.parse(json);
	}
	return new Layer(json);
}