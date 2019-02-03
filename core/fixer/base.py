class Fixer(object):

    description = "Override this in subclasses with a description of fixer behavior."

    def __init__(self, experiment):
        self.experiment = experiment

    def __call__(self, md):
        raise NotImplementedError("Please implement this in subclasses.")
