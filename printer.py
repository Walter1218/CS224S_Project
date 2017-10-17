import sys

class LoggingPrinter:

    def __init__(self, filename):
        self.out_file = open(filename, "w")
        self.old_stdout = sys.stdout
        # This object will take over `stdout`'s job
        sys.stdout = self

    # Executed when the user does a `print`
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)

    # Executed when `with` block begins
    def __enter__(self): 
        return self

    # Executed when `with` block ends
    def __exit__(self, type, value, traceback): 
        # We don't want to log anymore. Restore the original stdout object.
        self.out_file.close()
        sys.stdout = self.old_stdout