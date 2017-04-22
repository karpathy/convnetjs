import { Vol } from "./convnet_vol";
import { LayerOptions, ILayer, LayerJSON, ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";
import { LossLayerOptions, SVMLayer, RegressionLayer, SoftmaxLayer } from "./convnet_layers_loss";
import { DotproductsLayerOptions, FullyConnLayer, ConvLayer } from "./convnet_layers_dotproducts";
import { MaxoutLayer, TanhLayer, SigmoidLayer, ReluLayer } from "./convnet_layers_nonlinearities";
import { PoolLayer } from "./convnet_layers_pool";
import { InputLayer } from "./convnet_layers_input";
import { DropoutLayer } from "./convnet_layers_dropout";
import { LocalResponseNormalizationLayer } from "./convnet_layers_normalization";

const assert = util.assert;

export interface NetJSON{
    layers?: LayerJSON[];
}

/**
 * Net manages a set of layers
 * For now constraints: Simple linear order of layers, first layer input last layer a cost layer
 */
export class Net {
    layers: ILayer[];
    constructor(options?: LayerOptions[]) {
        if(!options){
            options = [];
        }
        this.layers = [];
    }
    // takes a list of layer definitions and creates the network layer objects
    makeLayers(defs: LayerOptions[]) {

        // few checks
        assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
        assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

        // desugar layer_defs for adding activation, dropout layers etc
        const desugar = function (defs: LayerOptions[]) {
            const new_defs = new Array<LayerOptions>();
            for (let i = 0; i < defs.length; i++) {
                const def = defs[i];

                if (def.type === 'softmax' || def.type === 'svm') {
                    const lossDef = <LossLayerOptions>def;
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    new_defs.push({ type: 'fc', num_neurons: lossDef.num_classes });
                }
                if (def.type === 'regression') {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    new_defs.push({ type: 'fc', num_neurons: def.num_neurons });
                }

                if (def.type === 'fc' || def.type === 'conv') {
                    const dotDef = <DotproductsLayerOptions>def;
                    if (typeof (dotDef.bias_pref) === 'undefined') {
                        dotDef.bias_pref = 0.0;
                        if (typeof dotDef.activation !== 'undefined' && dotDef.activation === 'relu') {
                            dotDef.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
                            // otherwise it's technically possible that a relu unit will never turn on (by chance)
                            // and will never get any gradient and never contribute any computation. Dead relu.
                        }
                    }
                }

                new_defs.push(def);

                if (typeof def.activation !== 'undefined') {
                    if (def.activation === 'relu') { new_defs.push({ type: 'relu' }); }
                    else if (def.activation === 'sigmoid') { new_defs.push({ type: 'sigmoid' }); }
                    else if (def.activation === 'tanh') { new_defs.push({ type: 'tanh' }); }
                    else if (def.activation === 'maxout') {
                        // create maxout activation, and pass along group size, if provided
                        const gs = def.group_size !== 'undefined' ? def.group_size : 2;
                        new_defs.push({ type: 'maxout', group_size: gs });
                    }
                    else { console.log('ERROR unsupported activation ' + def.activation); }
                }
                if (typeof def.drop_prob !== 'undefined' && def.type !== 'dropout') {
                    new_defs.push({ type: 'dropout', drop_prob: def.drop_prob });
                }

            }
            return new_defs;
        }
        defs = desugar(defs);

        // create the layers
        this.layers = [];
        for (let i = 0; i < defs.length; i++) {
            const def = defs[i];
            if (i > 0) {
                const prev = this.layers[i - 1];
                def.in_sx = prev.out_sx;
                def.in_sy = prev.out_sy;
                def.in_depth = prev.out_depth;
            }

            switch (def.type) {
                case 'fc': this.layers.push(new FullyConnLayer(def)); break;
                case 'lrn': this.layers.push(new LocalResponseNormalizationLayer(def)); break;
                case 'dropout': this.layers.push(new DropoutLayer(def)); break;
                case 'input': this.layers.push(new InputLayer(def)); break;
                case 'softmax': this.layers.push(new SoftmaxLayer(def)); break;
                case 'regression': this.layers.push(new RegressionLayer(def)); break;
                case 'conv': this.layers.push(new ConvLayer(def)); break;
                case 'pool': this.layers.push(new PoolLayer(def)); break;
                case 'relu': this.layers.push(new ReluLayer(def)); break;
                case 'sigmoid': this.layers.push(new SigmoidLayer(def)); break;
                case 'tanh': this.layers.push(new TanhLayer(def)); break;
                case 'maxout': this.layers.push(new MaxoutLayer(def)); break;
                case 'svm': this.layers.push(new SVMLayer(def)); break;
                default: console.log('ERROR: UNRECOGNIZED LAYER TYPE: ' + def.type);
            }
        }
    }

    // forward prop the network.
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    forward(V: Vol, is_training?: boolean) {
        if (typeof (is_training) === 'undefined') { is_training = false; }
        let act = this.layers[0].forward(V, is_training);
        for (let i = 1; i < this.layers.length; i++) {
            act = this.layers[i].forward(act, is_training);
        }
        return act;
    }

    getCostLoss(V: Vol, y: number | number[] | Float64Array | { [key: string]: number }): number {
        this.forward(V, false);
        const N = this.layers.length;
        const loss = <number>this.layers[N - 1].backward(y);
        return loss;
    }

    /**
     * backprop: compute gradients wrt all parameters
     */
    backward(y: number | number[] | Float64Array | { [key: string]: number }): number {
        const N = this.layers.length;
        const loss = <number>this.layers[N - 1].backward(y); // last layer assumed to be loss layer
        for (let i = N - 2; i >= 0; i--) { // first layer assumed input
            this.layers[i].backward();
        }
        return loss;
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        // accumulate parameters and gradients for the entire network
        const response = [];
        for (let i = 0; i < this.layers.length; i++) {
            const layer_reponse = this.layers[i].getParamsAndGrads();
            for (let j = 0; j < layer_reponse.length; j++) {
                response.push(layer_reponse[j]);
            }
        }
        return response;
    }
    getPrediction() {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        const S = this.layers[this.layers.length - 1];
        assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');
        if (S instanceof SoftmaxLayer) {
            const p = S.out_act.w;
            let maxv = p[0];
            let maxi = 0;
            for (let i = 1; i < p.length; i++) {
                if (p[i] > maxv) { maxv = p[i]; maxi = i; }
            }
            return maxi; // return index of the class with highest class probability
        }
        throw Error("to getPrediction, the last layer must be softmax");
    }
    toJSON(): NetJSON {
        const json: NetJSON = {};
        json.layers = [];
        for (let i = 0; i < this.layers.length; i++) {
            json.layers.push(this.layers[i].toJSON());
        }
        return json;
    }
    fromJSON(json: NetJSON) {
        this.layers = [];
        for (let i = 0; i < json.layers.length; i++) {
            const Lj = json.layers[i]
            const t = Lj.layer_type;
            let L: ILayer;
            if (t === 'input') { L = new InputLayer(); }
            if (t === 'relu') { L = new ReluLayer(); }
            if (t === 'sigmoid') { L = new SigmoidLayer(); }
            if (t === 'tanh') { L = new TanhLayer(); }
            if (t === 'dropout') { L = new DropoutLayer(); }
            if (t === 'conv') { L = new ConvLayer(); }
            if (t === 'pool') { L = new PoolLayer(); }
            if (t === 'lrn') { L = new LocalResponseNormalizationLayer(); }
            if (t === 'softmax') { L = new SoftmaxLayer(); }
            if (t === 'regression') { L = new RegressionLayer(); }
            if (t === 'fc') { L = new FullyConnLayer(); }
            if (t === 'maxout') { L = new MaxoutLayer(); }
            if (t === 'svm') { L = new SVMLayer(); }
            L.fromJSON(Lj);
            this.layers.push(L);
        }
    }
}
