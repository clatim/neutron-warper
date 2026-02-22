from omegaconf import OmegaConf


def read_problem(file):
    problem_def = OmegaConf.load(file)
    return problem_def

if __name__ == "__main__":
    pass


