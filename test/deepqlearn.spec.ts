import { deepqlearn } from "../src/index";
import * as Chai from "chai";

const expect = Chai.expect;
describe("deepqlearn", () => {
    describe("Can run a simple example", function () {
        beforeEach(function () {
        });

        it("get an optimal action from the learned policy", function () {
            const brainOpt = { start_learn_threshold: 100 };
            const brain = new deepqlearn.Brain(3, 2, brainOpt); // 3 inputs, 2 possible outputs (0,1)
            const state = [0, 0, 0];
            for (let k = 0; k < 1000; k++) {
                const action = brain.forward(state); // returns index of chosen action
                const reward = action === 1 ? 1.0 : 0.0; //give a reward for action 1 (no matter what state is)
                brain.backward(reward); // <-- learning magic happens here
                state[Math.floor(Math.random() * 3)] = Math.random(); // change state
            }
            brain.epsilon_test_time = 0.0; // don't make any more random choices
            brain.learning = false;
            // get an optimal action from the learned policy
            const input = [1, 1, 1];
            const chosen_action = brain.forward(input);
            console.log("chosen action after learning: " + chosen_action);
            // tanh are their own layers. Softmax gets its own fully connected layer.
            // this should all get desugared just fine.
            expect(chosen_action).to.equal(1);
        });
    });
});

