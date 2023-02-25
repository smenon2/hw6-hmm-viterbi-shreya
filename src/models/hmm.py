import numpy as np
class HiddenMarkovModel:
    """This class can be used to create an HMM 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """_summary_

        Args:
            observation_states (np.ndarray): the observation states possible, in the order to be used
            hidden_states (np.ndarray): the hidden states possible, in the order to be used
            prior_probabilities (np.ndarray): prior probabilities of hidden states in the order specified in the hidden_states list
            transition_probabilities (np.ndarray): transition probabilities [:,hidden_states[i]]
            emission_probabilities (np.ndarray): emission probabilities [hidden_states[i],:][:,observation_states[j]]
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities= prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities