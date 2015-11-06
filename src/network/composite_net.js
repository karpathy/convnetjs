"use strict";

var convnetjs = require('../../build/convnetjs');
//var workflow = require('./workflow.js');

// var assign = require('object-assign');

/**
 * CompositeNet consiste of prime_nets
 * 		multiple prime_nets can be connect to within a composite net
 * 		multiple composite net can also be connected 
 */
class CompositeNet{
	constructor(opt) {
		this.type = 'CompositeNet';
  		this.networkMap = []; //input vol map
		
		this.init();
	}
	
	init(){
		//this.workflow = workflow;
	}
	
	/**
	 * net : a convnetjs.net object
	 * 
	 */
	registerNetwork(net, inputVol, outputLength, id){
		this.networkMap[id] = {
			net: net,
			workflow: net.workflow,
			inputVol: inputVol,
			outputLength : outputLength,
			dependency:[]
		};
	}
	
	/**
	 * add networks that depends on others
	 * Note: a net work is able to depend on multiple other networks => other network's output correspond to part of its input
	 * Output[0] to [length-1] => input.w[inputStart] => ...
	 */
	registerDependentNetwork(net, inputVol, outputLength, id, dependentId, inputStart){
		var self = this;
		
		if(!this.networkMap[dependentId] || !this.networkMap[dependentId].workflow){
			return false; //dependent network or workflow does not exist
		}
		
		if(!this.networkMap[id]){
			this.registerNetwork(net, inputVol, outputLength, id);
		}else if(this.networkMap[id].dependency[dependentId]){
			return false; // the dependency already exist
		}
		
		this.networkMap[id].dependency[dependentId] = {
			inputStart: inputStart,
			connectCb: function(is_training){
				net.forward(self.networkMap[dependentId].net.act, is_training, inputStart);
			}
		}
		
		this.networkMap[dependentId].workflow.addChangeListener(
			self.networkMap[id].dependency[dependentId].connectCb
		);
		
		return true;
	}
	
	/**
	 * opposite of register network
	 */
	unRegisterDependentNetwork(id, dependentId){
		var self = this;
		
		if(!this.networkMap[id] || !this.networkMap[dependentId] || !this.networkMap[id].dependency[dependentId]){
			return true;
		}
		
		//unmount cb
		this.networkMap[dependentId].workflow.removeFireListener(
			self.networkMap[id].dependency[dependentId].connectCb
		);
		
		delete self.networkMap[id].dependency[dependentId];
		
		return true;
	}
	
	//TODO add toJson/fromJson
	
}

// //Exports
module.exports = CompositeNet;

