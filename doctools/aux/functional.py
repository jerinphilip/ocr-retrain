from functools import reduce

def compose(*fns):
    #fns = reversed(fns)
    identity = lambda x: x
    cb = lambda f, g: lambda *a, **kw: f(g(*a, **kw))
    return lambda *a, **kw: reduce(cb, fns, identity)(*a, **kw)

if __name__ == '__main__':
    add1 = lambda x: x+1
    mul2 = lambda x: 2*x
    print(compose(add1, mul2)(4))
