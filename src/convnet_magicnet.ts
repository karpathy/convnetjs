import * as util from "./convnet_util";
import { Vol } from "./convnet_vol";
import { Net, NetJSON } from "./convnet_net";
import { Trainer, TrainerOptions } from "./convnet_trainers";
import { LayerOptions } from "./layers";

// used utilities, make explicit local references
const randf = util.randf;
const randi = util.randi;
const maxmin = util.maxmin;
const randperm = util.randperm;
const weightedSample = util.weightedSample;
const getopt = util.getopt;
const arrUnique = util.arrUnique;


export interface MagicNetOptions {
    [key: string]: number | string;
    train_ratio?: number;
    num_folds?: number;
    num_candidates?: number;
    num_epochs?: number;
    ensemble_size?: number;
    batch_size_min?: number;
    batch_size_max?: number;
    l2_decay_min?: number;
    l2_decay_max?: number;
    learning_rate_min?: number;
    learning_rate_max?: number;
    momentum_min?: number;
    momentum_max?: number;
    neurons_min?: number;
    neurons_max?: number;
}

export interface MagicNetJSON{
    nets?: NetJSON[];
}

export interface Fold {
    train_ix: number[];
    test_ix: number[];
}

export interface Candidate {
    acc?: number[];
    accv?: number; // this will maintained as sum(acc) for convenience
    layer_defs?: LayerOptions[];
    trainer_def?: TrainerOptions;
    net?: Net;
    trainer?: Trainer;
}

/*
A MagicNet takes data: a list of convnetjs.Vol(), and labels
which for now are assumed to be class indeces 0..K. MagicNet then:
- creates data folds for cross-validation
- samples candidate networks
- evaluates candidate networks on all data folds
- produces predictions by model-averaging the best networks
*/
export class MagicNet {
    data: Vol[];
    labels: number[];
    train_ratio: number;
    num_folds: number;
    num_candidates: number;
    num_epochs: number;
    ensemble_size: number;
    batch_size_min: number;
    batch_size_max: number;
    l2_decay_min: number;
    l2_decay_max: number;
    learning_rate_min: number;
    learning_rate_max: number;
    momentum_min: number;
    momentum_max: number;
    neurons_min: number;
    neurons_max: number;
    folds: Fold[];
    candidates: Candidate[];
    evaluated_candidates: Candidate[];
    unique_labels: number[];
    iter: number;
    foldix: number;
    finish_fold_callback: () => void;
    finish_batch_callback: () => void;
    constructor(data?: Vol[], labels?: number[], opt?: MagicNetOptions) {
        if (!opt) {
            opt = {};
        }
        if (typeof data === 'undefined') { data = []; }
        if (typeof labels === 'undefined') { labels = []; }

        // required inputs
        this.data = data; // store these pointers to data
        this.labels = labels;

        // optional inputs
        this.train_ratio = getopt(opt, 'train_ratio', 0.7);
        this.num_folds = getopt(opt, 'num_folds', 10);
        this.num_candidates = getopt(opt, 'num_candidates', 50); // we evaluate several in parallel
        // how many epochs of data to train every network? for every fold?
        // higher values mean higher accuracy in final results, but more expensive
        this.num_epochs = getopt(opt, 'num_epochs', 50);
        // number of best models to average during prediction. Usually higher = better
        this.ensemble_size = getopt(opt, 'ensemble_size', 10);

        // candidate parameters
        this.batch_size_min = getopt(opt, 'batch_size_min', 10);
        this.batch_size_max = getopt(opt, 'batch_size_max', 300);
        this.l2_decay_min = getopt(opt, 'l2_decay_min', -4);
        this.l2_decay_max = getopt(opt, 'l2_decay_max', 2);
        this.learning_rate_min = getopt(opt, 'learning_rate_min', -4);
        this.learning_rate_max = getopt(opt, 'learning_rate_max', 0);
        this.momentum_min = getopt(opt, 'momentum_min', 0.9);
        this.momentum_max = getopt(opt, 'momentum_max', 0.9);
        this.neurons_min = getopt(opt, 'neurons_min', 5);
        this.neurons_max = getopt(opt, 'neurons_max', 30);

        // computed
        this.folds = []; // data fold indices, gets filled by sampleFolds()
        this.candidates = []; // candidate networks that are being currently evaluated
        this.evaluated_candidates = []; // history of all candidates that were fully evaluated on all folds
        this.unique_labels = arrUnique(labels);
        this.iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
        this.foldix = 0; // index of active fold

        // callbacks
        this.finish_fold_callback = null;
        this.finish_batch_callback = null;

        // initializations
        if (this.data.length > 0) {
            this.sampleFolds();
            this.sampleCandidates();
        }
    };
    // sets this.folds to a sampling of this.num_folds folds
    sampleFolds() {
        const N = this.data.length;
        const num_train = Math.floor(this.train_ratio * N);
        this.folds = []; // flush folds, if any
        for (let i = 0; i < this.num_folds; i++) {
            const p = randperm(N);
            this.folds.push({ train_ix: p.slice(0, num_train), test_ix: p.slice(num_train, N) });
        }
    }

    // returns a random candidate network
    sampleCandidate() {
        const input_depth = this.data[0].w.length;
        const num_classes = this.unique_labels.length;

        // sample network topology and hyperparameters
        const layer_defs = [];
        layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: input_depth });
        const nl = weightedSample([0, 1, 2, 3], [0.2, 0.3, 0.3, 0.2]); // prefer nets with 1,2 hidden layers
        for (let q = 0; q < nl; q++) {
            const ni = randi(this.neurons_min, this.neurons_max);
            const act = ['tanh', 'maxout', 'relu'][randi(0, 3)];
            if (randf(0, 1) < 0.5) {
                const dp = Math.random();
                layer_defs.push({ type: 'fc', num_neurons: ni, activation: act, drop_prob: dp });
            } else {
                layer_defs.push({ type: 'fc', num_neurons: ni, activation: act });
            }
        }
        layer_defs.push({ type: 'softmax', num_classes: num_classes });
        const net = new Net();
        net.makeLayers(layer_defs);

        // sample training hyperparameters
        const bs = randi(this.batch_size_min, this.batch_size_max); // batch size
        const l2 = Math.pow(10, randf(this.l2_decay_min, this.l2_decay_max)); // l2 weight decay
        const lr = Math.pow(10, randf(this.learning_rate_min, this.learning_rate_max)); // learning rate
        const mom = randf(this.momentum_min, this.momentum_max); // momentum. Lets just use 0.9, works okay usually ;p
        const tp = randf(0, 1); // trainer type
        let trainer_def;
        if (tp < 0.33) {
            trainer_def = { method: 'adadelta', batch_size: bs, l2_decay: l2 };
        } else if (tp < 0.66) {
            trainer_def = { method: 'adagrad', learning_rate: lr, batch_size: bs, l2_decay: l2 };
        } else {
            trainer_def = { method: 'sgd', learning_rate: lr, momentum: mom, batch_size: bs, l2_decay: l2 };
        }

        const trainer = new Trainer(net, trainer_def);

        const cand: Candidate = {};
        cand.acc = [];
        cand.accv = 0; // this will maintained as sum(acc) for convenience
        cand.layer_defs = layer_defs;
        cand.trainer_def = trainer_def;
        cand.net = net;
        cand.trainer = trainer;
        return cand;
    }

    // sets this.candidates with this.num_candidates candidate nets
    sampleCandidates() {
        this.candidates = []; // flush, if any
        for (let i = 0; i < this.num_candidates; i++) {
            const cand = this.sampleCandidate();
            this.candidates.push(cand);
        }
    }

    step() {

        // run an example through current candidate
        this.iter++;

        // step all candidates on a random data point
        const fold = this.folds[this.foldix]; // active fold
        const dataix = fold.train_ix[randi(0, fold.train_ix.length)];
        for (let k = 0; k < this.candidates.length; k++) {
            const x = this.data[dataix];
            const l = this.labels[dataix];
            this.candidates[k].trainer.train(x, l);
        }

        // process consequences: sample new folds, or candidates
        const lastiter = this.num_epochs * fold.train_ix.length;
        if (this.iter >= lastiter) {
            // finished evaluation of this fold. Get final validation
            // accuracies, record them, and go on to next fold.
            const val_acc = this.evalValErrors();
            for (let k = 0; k < this.candidates.length; k++) {
                const c = this.candidates[k];
                c.acc.push(val_acc[k]);
                c.accv += val_acc[k];
            }
            this.iter = 0; // reset step number
            this.foldix++; // increment fold

            if (this.finish_fold_callback !== null) {
                this.finish_fold_callback();
            }

            if (this.foldix >= this.folds.length) {
                // we finished all folds as well! Record these candidates
                // and sample new ones to evaluate.
                for (let k = 0; k < this.candidates.length; k++) {
                    this.evaluated_candidates.push(this.candidates[k]);
                }
                // sort evaluated candidates according to accuracy achieved
                this.evaluated_candidates.sort(function (a, b) {
                    return (a.accv / a.acc.length)
                        > (b.accv / b.acc.length)
                        ? -1 : 1;
                });
                // and clip only to the top few ones (lets place limit at 3*ensemble_size)
                // otherwise there are concerns with keeping these all in memory
                // if MagicNet is being evaluated for a very long time
                if (this.evaluated_candidates.length > 3 * this.ensemble_size) {
                    this.evaluated_candidates = this.evaluated_candidates.slice(0, 3 * this.ensemble_size);
                }
                if (this.finish_batch_callback !== null) {
                    this.finish_batch_callback();
                }
                this.sampleCandidates(); // begin with new candidates
                this.foldix = 0; // reset this
            } else {
                // we will go on to another fold. reset all candidates nets
                for (let k = 0; k < this.candidates.length; k++) {
                    const c = this.candidates[k];
                    const net = new Net();
                    net.makeLayers(c.layer_defs);
                    const trainer = new Trainer(net, c.trainer_def);
                    c.net = net;
                    c.trainer = trainer;
                }
            }
        }
    }

    evalValErrors() {
        // evaluate candidates on validation data and return performance of current networks
        // as simple list
        const vals = [];
        const fold = this.folds[this.foldix]; // active fold
        for (let k = 0; k < this.candidates.length; k++) {
            const net = this.candidates[k].net;
            let v = 0.0;
            for (let q = 0; q < fold.test_ix.length; q++) {
                const x = this.data[fold.test_ix[q]];
                const l = this.labels[fold.test_ix[q]];
                net.forward(x);
                const yhat = net.getPrediction();
                v += (yhat === l ? 1.0 : 0.0); // 0 1 loss
            }
            v /= fold.test_ix.length; // normalize
            vals.push(v);
        }
        return vals;
    }

    /**
     * returns prediction scores for given test data point, as Vol
     * uses an averaged prediction from the best ensemble_size models
     * x is a Vol.
     */
    predict_soft(data: Vol): Vol {
        // forward prop the best networks
        // and accumulate probabilities at last layer into a an output Vol

        let eval_candidates = [];
        let nv = 0;
        if (this.evaluated_candidates.length === 0) {
            // not sure what to do here, first batch of nets hasnt evaluated yet
            // lets just predict with current candidates.
            nv = this.candidates.length;
            eval_candidates = this.candidates;
        } else {
            // forward prop the best networks from evaluated_candidates
            nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
            eval_candidates = this.evaluated_candidates
        }

        // forward nets of all candidates and average the predictions
        let xout, n;
        for (let j = 0; j < nv; j++) {
            const net = eval_candidates[j].net;
            const x = net.forward(data);
            if (j === 0) {
                xout = x;
                n = x.w.length;
            } else {
                // add it on
                for (let d = 0; d < n; d++) {
                    xout.w[d] += x.w[d];
                }
            }
        }
        // produce average
        for (let d = 0; d < n; d++) {
            xout.w[d] /= nv;
        }
        return xout;
    }

    predict(data: Vol): number {
        const xout = this.predict_soft(data);
        let predicted_label:number;
        if (xout.w.length !== 0) {
            const stats = maxmin(xout.w);
            predicted_label = stats.maxi;
        } else {
            predicted_label = -1; // error out
        }
        return predicted_label;

    }

    toJSON() {
        // dump the top ensemble_size networks as a list
        const nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
        const json: MagicNetJSON = {};
        json.nets = [];
        for (let i = 0; i < nv; i++) {
            json.nets.push(this.evaluated_candidates[i].net.toJSON());
        }
        return json;
    }

    fromJSON(json: MagicNetJSON) {
        this.ensemble_size = json.nets.length;
        this.evaluated_candidates = [];
        for (let i = 0; i < this.ensemble_size; i++) {
            const net = new Net();
            net.fromJSON(json.nets[i]);
            const dummy_candidate: Candidate = {};
            dummy_candidate.net = net;
            this.evaluated_candidates.push(dummy_candidate);
        }
    }

    // callback functions
    // called when a fold is finished, while evaluating a batch
    onFinishFold(f: { (): void }) { this.finish_fold_callback = f; }
    // called when a batch of candidates has finished evaluating
    onFinishBatch(f: { (): void }) { this.finish_batch_callback = f; }

};
