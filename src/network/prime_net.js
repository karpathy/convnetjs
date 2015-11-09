"use strict";

var convnetjs = require('../../build/convnetjs');
var Workflow = require('./workflow.js');
var util = require('./util.js');

class AutoEncoder extends convnetjs.Net{
	constructor(opt) {
    	super();
		
		//required
		this.in_depth = opt.in_depth;
		
		//optional
		this.bufferSize = typeof opt.bufferSize !== 'undefined' ? opt.bufferSize : 1;
		
		this.activation = typeof opt.activation !== 'undefined' ? opt.activation : 'step';
		
		this.out_depth = typeof opt.out_depth !== 'undefined' ? opt.out_depth : this.in_depth;
		this.in_sx = typeof opt.in_sx !== 'undefined' ? opt.in_sx : 1;
		this.in_sy = typeof opt.in_sy !== 'undefined' ? opt.in_sy : 1;
		this.out_sx = typeof opt.out_sx !== 'undefined' ? opt.out_sx : this.in_sx;
		this.out_sy = typeof opt.out_sy !== 'undefined' ? opt.out_sy : this.in_sy;
		
		var hiddenNeuron = this.in_depth * this.in_sx * this.in_sy;
		var outputNeuron = this.out_depth * this.out_sx * this.out_sy;
		
		this.outputLength = outputNeuron;
		
		this.layer_defs = [];
		this.layer_defs.push({type:'input', out_sx:this.in_sx, out_sy:this.in_sy, out_depth:this.in_depth});
		this.layer_defs.push({type:'fc', num_neurons:this.bufferSize, bias_pref: 1.0, activation: this.activation});
		this.layer_defs.push({type:'buffer', bufferSize:hiddenNeuron});
		this.layer_defs.push({type:'regression', num_neurons:outputNeuron});
		
		this.makeLayers(this.layer_defs);
		this.type = 'Autoencoder';
  		
		this.init();
	}
	
	init(){
		this.inputVol = new convnetjs.Vol(this.in_sx, this.in_sy, this.in_depth, 0);
		this.trainer = new convnetjs.Trainer(this, {learning_rate:0.5, method:'adadelta', batch_size:25, l2_decay:0.00001, l1_decay:0.0});
		this.workflow = new Workflow();
	}
	
	forward(is_training, V, startInd){
		if(V && V.w){
			util.smartCopy(V, this.inputVol, startInd); //copy values
		}
		
		this.is_training = is_training;
		this.act = super.forward(this.inputVol,is_training);
		this.workflow.emitFire(is_training);
		return this.act;
	}
	  
	train(x,y){
		if(!x){
			return;
		}
		var inputVol = x;
		var targetValue = (!y) ? x.w : y; //default same output as the input
			
		return this.trainer.train(inputVol, targetValue);
	}
	
	toJSON() {
      var json = super.toJSON();
	  json.type = this.type;
      return json;
    }
	
	fromJSON(json) {
		super.fromJSON(json);
		this.type = json.type;
		this.init();
	}
}

// //Exports
module.exports.AutoEncoder = AutoEncoder;

