# Trajectory Optimization of Multirotors through Gates in Drone Racing using RL


This project is an attempt to use policy gradient based methods to optimize a trajectory through gates for use in drone racing. The primary endpoint is measured by reduction in snap of the trajectory while keeping it feasible.


### TODO
 - Implement Baseline
 - Implement learning algorithm
 - Train and get results
 - Write report

### Stretch
 - Modify environment to do gradient descent based segment time optimization
 
### DONE
 - Create the environment 
   - Define Gym environment
   - Create reward function
 - Decide with RL algorithm to use
   - Decided to go with SAC
 - Decide NN architecture
   - Trivial architectures for now