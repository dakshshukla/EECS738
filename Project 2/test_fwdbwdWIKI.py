# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:19:10 2019

@author: Daksh
"""

states = ('Healthy', 'Fever')
end_state = 'E'
 
observations = ('normal', 'cold', 'dizzy')
 
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
 
transition_probability = {
   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
   }
 
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }

fwd = []
f_prev = {}
for i, observation_i in enumerate(observations):
    f_curr = {}
    for st in states:
        if i == 0:
            # base case for the forward part
            prev_f_sum = start_probability[st]
        else:
            prev_f_sum = sum(f_prev[k]*transition_probability[k][st] for k in states)

        f_curr[st] = emission_probability[st][observation_i] * prev_f_sum

    fwd.append(f_curr)
    f_prev = f_curr

p_fwd = sum(f_curr[k] * transition_probability[k][end_state] for k in states)

# backward part of the algorithm
bkw = []
b_prev = {}
for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
    b_curr = {}
    for st in states:
        if i == 0:
            # base case for backward part
            b_curr[st] = transition_probability[st][end_state]
        else:
            b_curr[st] = sum(transition_probability[st][l] * emission_probability[l][observation_i_plus] * b_prev[l] for l in states)

    bkw.insert(0,b_curr)
    b_prev = b_curr

p_bkw = sum(start_probability[l] * emission_probability[l][observations[0]] * b_curr[l] for l in states)

# merging the two parts
posterior = []
for i in range(len(observations)):
    posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
