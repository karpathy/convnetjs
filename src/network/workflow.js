"use strict";
var EventEmitter = require('events').EventEmitter;
var assign = require('object-assign');  
var FIRE_EVENT = 'FIRE';
var DRILL_EVENT = 'DRILL'; //is_training = true

var workflow = assign({}, EventEmitter.prototype, {
  emitFire: function(is_training) {
      if(is_training){
        this.emit(FIRE_EVENT);
      }else{
        this.emit(DRILL_EVENT);
      }
  },
  
  emitDrill: function() {
      this.emit(DRILL_EVENT);
  },
  
  /**
  * @param {function} callback (is_training)
  */
  addFireListener: function(callback) {
      this.on(FIRE_EVENT, callback(false));
      this.on(DRILL_EVENT, callback(true));
  },

  /**
  * @param {function} callback
  */
  removeFireListener: function(callback) {
      this.removeListener(FIRE_EVENT, callback);
      this.removeListener(DRILL_EVENT, callback);
  }
});

module.exports = workflow;
