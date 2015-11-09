"use strict";
var EventEmitter = require('events').EventEmitter;
var FIRE_EVENT = 'FIRE';
var DRILL_EVENT = 'DRILL'; //is_training = true

class workflow extends EventEmitter {
  constructor() {
    	super();  
  }
  
  emitFire(is_training) {
      if(is_training){
        this.emit(FIRE_EVENT);
      }else{
        this.emit(DRILL_EVENT);
      }
  }
  
  emitDrill() {
      this.emit(DRILL_EVENT);
  }
  
  /**
  * @param {function} callback (is_training)
  */
  addFireListener(callback) {
      this.on(FIRE_EVENT, function(){
        callback(true)
      });
      this.on(DRILL_EVENT, function(){
        callback(false);
      });
  }

  /**
  * @param {function} callback
  */
  removeFireListener(callback) {
      this.removeListener(FIRE_EVENT, callback);
      this.removeListener(DRILL_EVENT, callback);
  }
};

module.exports = workflow;
