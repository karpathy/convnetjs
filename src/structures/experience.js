// An agent is in state0 and does action0
// environment then assigns reward0 and provides new state, state1
// Experience nodes store all this information, which is used in the
// Q-learning update step

export default class Experience {
	constructor(state0, action0, reward0, state1){
		this.state0 = state0;
		this.action0 = action0;
		this.reward0 = reward0;
		this.state1 = state1;
	}
}