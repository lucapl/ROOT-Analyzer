
class Event():

    def __init__(self,limit=48):
        self.timer = 0
        self.msg = ""
        self.limit = 48

    def over_limit(self):
        return self.timer > self.limit
    
    def update(self):
        if not self.over_limit():
            self.timer += 1
    
    def reset(self):
        self.timer = 0

    def get(self):
        if self.over_limit():
            return ""
        else:
            return self.msg
