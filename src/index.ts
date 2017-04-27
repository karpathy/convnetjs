// direct export
export { Vol } from "./convnet_vol";
export { Net } from "./convnet_net";
export { MagicNet } from "./convnet_magicnet";
export { randf, randi, randn, randperm } from "./convnet_util";
export { ConvLayer } from "./convnet_layers_dotproducts";

// module export
import * as cnnvis from "./cnnvis";
import * as cnnutil from "./cnnutil";
import * as util from "./convnet_util";
import * as volutil from "./convnet_vol_util";
import * as deepqlearn from "./deepqlearn";

export { cnnvis };
export { cnnutil };
export { util };
export { volutil };
export { deepqlearn };

// rename
import { Trainer } from "./convnet_trainers";
export { Trainer, Trainer as SGDTrainer };
