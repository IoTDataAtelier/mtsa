class ParameterError(Exception):

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message