import { Vol, VolJSON } from "./convnet_vol";
export interface LayerOptions {
    [key: string]: number | string;
    type: string;
}

export interface LayerJSON {
    [key: string]: number | string | number[] | VolJSON | VolJSON[];
    filters?: VolJSON[];
}

export interface ParamsAndGrads {
    params: number[] | Float64Array;
    grads: number[] | Float64Array;
    l2_decay_mul: number;
    l1_decay_mul: number;
}


export interface ILayer<T extends string> {
    layer_type: T;
    in_sx: number;
    in_sy: number;
    in_depth: number;
    out_sx: number;
    out_sy: number;
    out_depth: number;
    forward(V: Vol, is_training: boolean): Vol;
    backward(y?: number | number[] | Float64Array | { [key: string]: number }): void | number;
    getParamsAndGrads(): ParamsAndGrads[];
    toJSON(): LayerJSON;
    fromJSON(json: LayerJSON): void;
}

export class LayerBase<T extends string> {
    layer_type: T;
    in_sx: number;
    in_sy: number;
    in_depth: number;
    out_sx: number;
    out_sy: number;
    out_depth: number;
    constructor(layerType: T, opt?: LayerOptions) {
        this.layer_type = layerType;
        if (!opt) { return; }
    }
}
