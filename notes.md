# Notes of how this repository works

## How to use the pettingzoo hunt environment

it actually runs the gym hunt environment under the hood, it sets the self.env to the gym env

[gym_stag_hunt/envs/pettingzoo/shared.py](gym_stag_hunt/envs/pettingzoo/shared.py) has the pettingzoo parallel env, 
and implements the basic pettingzoo API

[gym_stag_hunt/envs/gym/hunt.py](gym_stag_hunt/envs/gym/hunt.py) is the gym env, it doesn't do much, it sets the Game as Stag Hunt

[gym_stag_hunt/src/games/staghunt_game.py](gym_stag_hunt/src/games/staghunt_game.py) the real game, where all the logic happens, 
inherits from abstract grid game

[gym_stag_hunt/src/games/abstract_grid_game.py](gym_stag_hunt/src/games/abstract_grid_game.py) the abstract grid game class

the code seems unnecessarily complicated, the things are split between files, presumably because it also needs to render the pygame things, 
and with environment inside environment, the performance isn't that good too

but a benefit of keeping this code is that it already has the pygame set up, and eye tracking has been done, 
so continue using this makes it easier to integrate that research