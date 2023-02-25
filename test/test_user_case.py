"""
UCSF BMI203: Biocomputing Algorithms
Author: Shreya Menon
Date: 02/24/23
Program: BMI
Description: HW6 Implementation of Viterbi Algorithm
"""
import pytest
import numpy as np
from models.hmm import HiddenMarkovModel
from models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """This is test case that use a HMM model and states described in lecture
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('../data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    # These are tests I added
    assert len(use_case_one_hmm.observation_states) == 2
    assert len(use_case_one_hmm.hidden_states) == 2
    assert len(use_case_one_viterbi.hmm_object.hidden_states) == 2
    assert len(use_case_one_viterbi.hmm_object.observation_states) == 2
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """This is a test case that uses HMM object and data stored in UserCase-one
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('../data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    assert len(use_case_one_hmm.observation_states) == 2
    assert len(use_case_one_hmm.hidden_states) == 2
    assert len(use_case_one_viterbi.hmm_object.hidden_states) == 2
    assert len(use_case_one_viterbi.hmm_object.observation_states) == 2
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_two():
    """This is a test case with data that is supposed to model the weather in the locations. The observed states are "Mission Bay" or "Parnassus".
    The hidden states are "Hot" or "Cold". We can ask the question: can we decode the weather from an observed sequence of locations using an HMM?
    I hypothesize that we can decode the weather, and that cold hidden states will be associated with being at Parnassus. 
    """
    hidden_states = ['Mission Bay','Parnassus']
    observation_states = ['Cold','Hot']
    prior_probability = np.array([0.9, 0.1])
    transition_probabilities = np.array([[0.1,0.9],[0.2,0.8]])
    emission_probabilities = np.array([[0.1,0.9],[0.2,0.8]])
    hmm = HiddenMarkovModel(observation_states,
                                         hidden_states, prior_probability, transition_probabilities, emission_probabilities)
    viterbi = ViterbiAlgorithm(hmm)
    best = viterbi.best_hidden_state_sequence(['Cold','Cold','Cold','Cold'])
    assert np.alltrue(best==np.array(['Mission Bay', 'Parnassus', 'Parnassus', 'Parnassus'])), "The test case isn't finding the expected hidden states"


def test_user_case_three():
    """This is another test case with a fictional data. The observed states are "Tired" or "Rested". The hidden states are "Cold" or "Warm". 
    We can ask the question: can we decode a sequence of hidden states of whether a student is cold or warm from a sequence of observed states of tired or rested using an HMM?
    I hypothesize that we can decode the hidden states, and that warm will be associated with being rested. 
    """
    hidden_states = ['Cold','Warm']
    observation_states = ['Rested','Tired']
    prior_probability = np.array([0.1, 0.9])
    transition_probabilities = np.array([[0.1,0.9],[0.2,0.8]])
    emission_probabilities = np.array([[0.1,0.9],[0.2,0.8]])
    hmm = HiddenMarkovModel(observation_states,
                                         hidden_states, prior_probability, transition_probabilities, emission_probabilities)
    viterbi = ViterbiAlgorithm(hmm)
    best = viterbi.best_hidden_state_sequence(['Rested','Rested','Rested','Rested'])
    assert np.alltrue(best==np.array(['Warm', 'Warm', 'Warm', 'Warm'])), "The test case isn't finding the expected hidden states"
