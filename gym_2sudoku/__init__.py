from gym.envs.registration import register

register(
    id='sudoku2-v0',
    entry_point='gym_2sudoku.envs:Sudoku2Env',
)
