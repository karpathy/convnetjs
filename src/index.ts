// direct export
export {Vol} from "./convnet_vol";
export {Net} from "./convnet_net";

// module export
import * as cnnvis from "./cnnvis";
import * as cnnutil from "./cnnutil";
import * as util from "./convnet_util";
import * as deepqlearn from "./deepqlearn";

export {util};
export {cnnvis};
export {cnnutil};
export {deepqlearn};

// rename
import {Trainer} from "./convnet_trainers";
export {Trainer, Trainer as SGDTrainer};
