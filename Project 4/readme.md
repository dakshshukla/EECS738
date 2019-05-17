Project 4 - EECS 738 Spring 2019

This project consists of 2 files/ codes:

1. maze_1.py which is based on the following post
	https://www.samyzaf.com/ML/rl/qmaze.html
	
2. treasure_hunter.py - This is the main code for solving a maze/ treasure hunt.
The hunter starts from top left corner cell, avoids some obstacles and the final goal is to
reach a cell in the bottom right corner.

__________________________________________________________________________________________________________


treasure_hunter.py

This is python 3.7 code, to run: python treasure_hunter.py

1. The code first creates a 10 by 10 cell grid with white spaces representing free cells,
black cells representing obstacles, gray cells with 0.5 value represents where hunter currently is,
and gray cells with 0.8 value (light gray) represents previously visited cell.

2. The code contains a neural network written from scratch, with 1 hidden layer containing
100 neurons, that is used for training and it behaves as the agent. The code also contains (commented out)
keras model just to test the algorithm.

3. Input to neural network is a 101 by 1 row vector that contains flattened out 10 by 10 grid with
grayscale color values (states) and the last column (101st) is the action taken in that state. The output
is the total value/ quality number to take that action in that state and accouted for Bellman's equation.

4. The main algorithm uses Q-learning algorithm, based on course slides:

Q = np.max(q) + beta*( reward + gamma*max_Qa - np.max(q) )

5. The code builds up data for training by 10% exploration and 90% exploitation.

6. It also visualizes states of the map grig in a image show plot after the run of each episode,
and gradually the progress of the agent training can be monitered with the help of this plot.
