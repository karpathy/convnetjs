"use strict";

/**
* copy all values from v1 to v2 starting from v2[startInd]
*/
module.exports.smartCopy = function(v1, v2, startInd){
	if(!startInd){
		startInd = 0;
	}
	
	var v1w = v1.getW();
	var v2w = v2.getW();
	
  	for(var i = 0; i < v1w.length; i++){
		var ind = startInd + i;
		if(ind >= v2w.length){
		break;
		}
		v2w[ind] = v1w[i];
	}
	return v2;
};
