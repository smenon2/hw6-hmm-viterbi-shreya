import copy
import numpy as np


class ViterbiAlgorithm:
    """This class can be used to run Viterbi Algorithm on an instance of HMM to find the most likely sequence of hidden states given a sequence of observed states
    """

    def __init__(self, hmm_object):
        """This is creates an instance of Viterbi Algorithm

        Args:
            hmm_object (HiddenMarkovModel): an instance of HiddenMarkovModel
        """
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): an array of a sequence of observed states that are going to be decoded using a Viterbi algorithm

        Returns:
            np.ndarray: an array of the most likely sequence of hidden states given a sequence of observed states and HMM 
        """

        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states),
                         len(self.hmm_object.hidden_states)))
        path[0, :] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]

        best_path = np.zeros((len(decode_observation_states),
                              len(self.hmm_object.hidden_states)))

        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        # 2. Scale      
        state_0 = self.hmm_object.observation_states_dict[decode_observation_states[0]]
        delta = np.multiply(self.hmm_object.prior_probabilities, self.hmm_object.emission_probabilities[:, state_0])
        delta = delta / np.sum(delta)

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):

            # TODO: comment the initialization, recursion, and termination steps

            product_of_delta_and_transition_emission = np.multiply(delta,
                                                                   self.hmm_object.transition_probabilities.transpose())

            # Update delta and scale

            # Select the hidden state sequence with the maximum probability

            # Update best path

            product_of_delta_and_transition_emission = np.multiply(delta,
                                                                   self.hmm_object.transition_probabilities.transpose())
            max_pointer = np.argmax(product_of_delta_and_transition_emission.transpose(), axis=0)
            max_prob = np.max(product_of_delta_and_transition_emission.transpose(), axis=0)

            state = self.hmm_object.observation_states_dict[decode_observation_states[trellis_node]]
            path[trellis_node, :] = max_pointer
            delta = np.multiply(max_prob, self.hmm_object.emission_probabilities[:, state])
            delta = delta / np.sum(delta)

            for hidden_state in range(len(self.hmm_object.hidden_states)):
                if hidden_state == max_pointer[np.argmax(delta)]:
                    best_path[trellis_node - 1, hidden_state] = 1
                else:
                    best_path[trellis_node - 1, hidden_state] = 0

            # Set best hidden state sequence in the best_path np.ndarray THEN copy the best_path to path
            path = best_path.copy()

        # Select the last hidden state, given the best path (i.e., maximum probability)

        for hidden_state in range(len(self.hmm_object.hidden_states)):
            if hidden_state == np.argmax(delta):
                best_path[-1, hidden_state] = 1
            else:
                best_path[-1, hidden_state] = 0

        decode_hidden = []
        for node in range(0, len(decode_observation_states)):
            idx = np.argmax(best_path[node, :])
            decode_hidden.append(self.hmm_object.hidden_states_dict[idx])

        best_hidden_state_path = np.array(decode_hidden)

        return best_hidden_state_path
