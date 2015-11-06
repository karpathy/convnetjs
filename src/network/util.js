"use strict";

/**
* copy all values from v1 to v2 starting from v2[startInd]
*/
module.exports.smartCopy = function(v1, v2, startInd){
	if(!startInd){
		startInd = 0;
	}
	
  	for(var i = 0; i < v1.w.length; i++){
		var ind = startInd + i;
		if(ind >= v2.w.length){
		break;
		}
		v2.w[ind] = v1.w[i];
	}
	return v2;
};
