# AI_Snake

Self-project with the goal of continued learning and using of neural network architecture.

A base Snake game was created using Pygame with hopes of first adding a network to train the model used to predict which direction the snake should proceed.

~~Further goals include incorporating a genetic algorithm.~~
 
 Genetic Algorithm achieved! Also incorporated a visualization of the neural network firing as it plays. Can save weights and load rather than retraining the AI each playthrough.
 
 The network is a feed forward network and works well with an input layer of 7 nodes (left_blocked, front_blocked, right_blocked, normalized_angle_to_food(x/y), normalized direction vector(x/y)), 2 hidden layers with 9 and 15 nodes, and of course an output layer of 3 (left, straight, right). All this is visualized on screen. 
 
 Next is adding/changing inputs to modify behavior/performance? Maybe give it some sort of angle to food, given the A* shortest distance to the food... TBD
 
 GIF of the A.I. at work:
 https://i.imgur.com/2zTvM44.gifv
