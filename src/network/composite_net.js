"use strict";

var convnetjs = require('../../build/convnetjs');
var Workflow = require('./workflow.js');

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
		this.workflow = new Workflow();
	}
	
	/**
	 * fire a particular network
	 */
	fireNetwork(id, is_training, V, startInd){
		if(!this.networkMap[id]){
			return 'network does not exist';
		}
		return this.networkMap[id].net.forward(is_training, V, startInd);
	}
	
	/**
	 * net : a convnetjs.net object
	 * 
	 */
	registerNetwork(net, id){
		if(!net.workflow || !net.inputVol || !net.outputLength){
			return "workflow, inputVol or outputLength missing";
		}
		
		this.networkMap[id] = {
			net: net,
			workflow: net.workflow,
			inputVol: net.inputVol,
			outputLength : net.outputLength,
			outputListener: [],
			dependency:[]
		};
		
		return null; //error is null
	}
	
	/**
	 * TODO
	 */
	registerInputAction(){
		
	}
	
	/**
	 * cb (is_training) is call back function
	 */
	registerOutputListener(cb, id){
		if(typeof cb !== "function"){
			return 'callback must be function';
		}
		
		if(!this.networkMap[id]){
			return 'network does not exist';
		}
		
		this.networkMap[id].outputListener.push(cb);
		this.networkMap[id].workflow.addFireListener(cb);
		
		return null;
	}
	
	/**
	 * add networks that depends on others
	 * Note: a net work is able to depend on multiple other networks => other network's output correspond to part of its input
	 * Output[0] to [length-1] => input.w[inputStart] => ...
	 */
	registerDependentNetwork(net, id, dependentId, inputStart){
		var self = this;
		
		if(!this.networkMap[dependentId]){
			return "dependent network does not exist"; //dependent network or workflow does not exist
		}
		
		if(!this.networkMap[dependentId].workflow){
			return "dependent workflow does not exist"; //dependent network or workflow does not exist
		}
		
		if(!this.networkMap[id]){
			this.registerNetwork(net, id);
		}
		
		this.networkMap[id].dependency[dependentId] = {
			inputStart: inputStart,
			connectCb: function(is_training){
				net.forward(is_training, self.networkMap[dependentId].net.act, inputStart);
			}
		}
		
		this.networkMap[dependentId].workflow.addFireListener(
			self.networkMap[id].dependency[dependentId].connectCb
		);
		
		return null;
	}
	
	/**
	 * TODO: opposite of register network
	 */
	// unRegisterDependentNetwork(id, dependentId){
	// 	var self = this;
		
	// 	if(!this.networkMap[id] || !this.networkMap[dependentId] || !this.networkMap[id].dependency[dependentId]){
	// 		return true;
	// 	}
		
	// 	//unmount cb
	// 	this.networkMap[dependentId].workflow.removeFireListener(
	// 		self.networkMap[id].dependency[dependentId].connectCb
	// 	);
		
	// 	delete self.networkMap[id].dependency[dependentId];
		
	// 	return true;
	// }
	
	//TODO add toJson/fromJson
	
}

// //Exports
module.exports = CompositeNet;

