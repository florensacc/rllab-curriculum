from sandbox.haoran.hashing.bonus_trpo.terminators.base import Terminator

class MontezumaTerminator(Terminator):
    def __init__(self,task,verbose=False):
        self.task = task
        self.verbose = verbose

    def set_env(self,env):
        self.env = env

    def is_terminal(self):
        """ can be accelerated by passing ram from env """
        if self.task == "to_second_room":
            ram = self.env.ale.getRAM()
            task_complete = ram[3] == 2
        elif self.task == "to_right_platform":
            ram = self.env.ale.getRAM()
            x = ram[42]
            y = ram[43]
            task_complete = (x > 104) and (y == 235)
        else:
            raise NotImplementedError

        if task_complete:
            message = "Terminate due to task %s complete"%(self.task)
            terminal = True
        else:
            if self.env.avoid_life_lost:
                if self.env.ale.game_over() or self.lives_lost:
                    message = "Terminate due to life loss"
                    terminal = True
                else:
                    terminal = False
            else:
                if self.env.ale.game_over():
                    message = "Terminate due to gameover"
                    terminal = True
                else:
                    terminal = False

        if self.verbose:
            print(message)


        return terminal
