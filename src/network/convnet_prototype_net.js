"use strict";

var convnetjs = require('../../build/convnetjs');
// var assign = require('object-assign');

class AutoEncoder extends convnetjs.Net{
	constructor(opt) {
    	super();
		
		//required
		this.in_depth = opt.in_depth;
		
		//optional
		this.out_depth = typeof opt.out_depth !== 'undefined' ? opt.out_depth : this.in_depth;
		this.in_sx = typeof opt.in_sx !== 'undefined' ? opt.in_sx : 1;
		this.in_sy = typeof opt.in_sy !== 'undefined' ? opt.in_sy : 1;
		this.out_sx = typeof opt.out_sx !== 'undefined' ? opt.out_sx : 1;
		this.out_sy = typeof opt.out_sy !== 'undefined' ? opt.out_sy : 1;
		
		var hiddenNeuron = this.in_depth * this.in_sx * this.in_sy;
		var outputNeuron = this.out_depth * this.out_sx * this.out_sy;
		
		this.layer_defs = [];
		this.layer_defs.push({type:'input', out_sx:this.in_sx, out_sy:this.in_sy, out_depth:this.in_depth});
		this.layer_defs.push({type:'fc', num_neurons:hiddenNeuron, bias_pref: 1.0, activation:'tanh'});
		this.layer_defs.push({type:'fc', num_neurons:hiddenNeuron, bias_pref: 1.0, activation:'step'});
		this.layer_defs.push({type:'regression', num_neurons:outputNeuron});
		
		this.makeLayers(this.layer_defs);
		this.type = 'Autoencoder';
  	}
}

// //Exports
module.exports.AutoEncoder = AutoEncoder;

