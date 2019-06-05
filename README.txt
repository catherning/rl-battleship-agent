Implementation of RL agents playing Battleships game

Code based on ideas described in DeepMind's paper about state-of-the-art RL net Alpha Zero,
playing against itself alternating Markov Games

Ships layout placement not considered in this task, because separate NN can be trained to place ships efficiently
Here focus is on the Agent that is hunting for floating ships

Agent is given with array where hits are described, missing hit(water) are -1, success(ship hit) are 1, everything else is 0 (fog)
so rest of the floating ships are hiding and the goal of the agent is to learn strategy how to predict floating ships location

Has been evaluated three networks:
1. Res-Net19, original as described in paper, having 19 residual blocks
2. Res-Net3, having only 3 residual blocks
3. Regular CNN as feature extractor with two fully-connected layers as classifiers for actions probability an states values respectively

Training speed depends most on:
1. MCTS playout number
2. Field size


Performance and training speed for 4 iterations (can be enough for qualitative summary)

Res-Net19: seems performing better than others, loss going down, but training a little slower than CNN because of bigger new size

Training Summary for 5 iterations:
loss:               4.2533 improved to 1.7123
policy_head_loss:   3.5906 improved to 1.1114
value_head_loss:    0.6627 improved to 0.6009


Res-Net3: seems performing poor, because at every iteration loss starts from about initial values and then going down to near constant value, I see no performance gain with number of iterations. 
My assumption that such jumps of loss and no progress are due to lack of feature-extractor power

Training Summary for 4 iterations:
loss:               4.7260 improved to 2.0847
policy_head_loss:   3.6656 improved to 1.4950
value_head_loss:    1.0604 improved to 0.5896


Regular CNN: training a bit faster comparing to Res-Net19, performance seems not so good as Res-Net19, but there is no results for longer training cycle and different hyper-parameters

Training Summary for 4 iterations:
loss:               4.7134 improved to 1.9586
policy_head_loss:   3.7671 improved to 1.2749
value_head_loss:    0.9463 improved to 0.6836


Any net performs poor or randomly if MCTS provided with small number of payouts or net given small amount of training examples

Crucial thing here is correct implementation of DeepMind's version of Monte-Carlo Tree Search
After about half day of debugging and checking things, it turns out that proper implementation cannot be done in a short term, 
so MCTS was cloned from here:
https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py


___________________________________________________________________

One shot from arbitrary 16x16 play

Ships array given for transparency
RL Agent given only Hits array

Legend: 
o - floating(alive and invisible) cell
x - drown
* - missing, water hit
@ - success, ship hit

Ships                             | Hits 
| - - - - - - - - - - - - - - - - |* - - * - - - * * - - * - * - - |
| - - - - - o o x - - x - - - - - |- - - - - - - @ - * @ - - - - - |
| - - - - - - - - - - x - - - - - |- - - - - - - - - - @ - - - - - |
| - - - - - - o - - - x - - - o - |- - * - - - - * * - @ - - - - - |
| - - - - - - o - - - o - o - o - |* - - - - - - - - - - - - * - - |
| - - - - - - - - - - - - o - o - |- - - - - - * - - - - - - * - - |
| - - - - - - - - - - - - o - - - |* - - * - * - * - - - - - * - - |
| - - - o o - - - - - - - x - - - |- - * - - - - * - - - - @ - - * |
| - - - - - - x x o - - - - - - - |* - - - - - @ @ - * - - - * - - |
| - - - - - - - - - - - - - - - - |- - * - - - - * - - - - - - - - |
| - - - - - - - - - - - o - - - - |- - * * - * - - - - - - - - - - |
| - - - - - - o o - - - o - - - - |- - - - - - - - * - - - - - - * |
| - - - - - - - - - - - - - - - - |* - - - * - - * - - - - - - - - |
| - - - - - - - - - - - - - - - - |- - - - - - - - - - - - - - - - |
| - o x o o o - - - - - - - - - - |- - @ - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - - |- - - * - - - - - - - - - - * * |