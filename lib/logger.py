from requests.models import Response
from .settings import DEBUG

# Python program to print
# colored text and background
def prRed(skk): return ("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): return ("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): return ("\033[93m{}\033[00m" .format(skk))
def prBlue(skk): return ("\033[94m{}\033[00m" .format(skk))
def prPurple(skk): return ("\033[95m{}\033[00m" .format(skk))
def prCyan(skk): return ("\033[96m{}\033[00m" .format(skk))
def prLightGray(skk): return ("\033[97m{}\033[00m" .format(skk))
def prBlack(skk): return ("\033[98m{}\033[00m" .format(skk))

class Logger:
    def info(self, data: str, prepend: str = ""):
        return print(prepend + prGreen("[INFO] ") + data)

    def warn(self, data: str, prepend: str = ""):
        return print(prepend + prYellow("[WARN] ") + data)

    def error(self, data: str, prepend: str = ""):
        return print(prepend + prRed("[ERROR] ") + data)

    def object(self, data: str):
        return print(prCyan("[OBJ] ") + data)

    def event(self, data: str):
        return print(prPurple("[EVENT] ") + data)

    def api(self, r: Response):
        if DEBUG:
            status = (prGreen if r.ok else prRed)("OK" if r.ok else "ERR")
            path = prLightGray(r.request.path_url)
            data = r.json()
            return print("{} {} {} {}".format(prBlue("[API]"), path, status, str(data)))

logger = Logger()
