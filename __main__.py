from lekit.Internal import *

class test:
    def call(self, a):
        print(a)

if __name__ == "__main__":
    #print(sys.argv)
    a = test()
    call = getattr(a, "call")
    call(50)