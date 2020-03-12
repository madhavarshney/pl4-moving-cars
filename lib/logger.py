# Python program to print
# colored text and background
def prRed(skk): return ("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): return ("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): return ("\033[93m{}\033[00m" .format(skk))
def prLightPurple(skk): return ("\033[94m{}\033[00m" .format(skk))
def prPurple(skk): return ("\033[95m{}\033[00m" .format(skk))
def prCyan(skk): return ("\033[96m{}\033[00m" .format(skk))
def prLightGray(skk): return ("\033[97m{}\033[00m" .format(skk))
def prBlack(skk): return ("\033[98m{}\033[00m" .format(skk))

class Logger:
    def info(self, data, prepend = ""):
        return print(prepend + prGreen("[INFO] ") + data)

    def warn(self, data):
        return print(prYellow("[WARN] ") + data)

    def error(self, data):
        return print(prRed("[ERROR] ") + data)

    def object(self, data):
        return print(prCyan("[OBJ] ") + data)

    def event(self, data):
        return print(prPurple("[EVENT] ") + data)

