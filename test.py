from multiprocessing import Pool

y = 0

def f(x):
    print(x)
    print(y)
    return (x[0] * x[0]) +x[1]


if __name__ == "__main__":
    pool = Pool()
    x = pool.apply_async(f,enumerate([1,2,3,2,1,6,5]))

    print(x)
