import time


def timer(*args):
    print(args, args[0])
    def decorator(func):
        def wrapper(*args1, **kwargs):
            start_time = time.time()
            result = func()
            end_time = time.time()
            if end_time - start_time > args[0]:
                print(f'{func.__name__} took too longer time')
            return result
        return wrapper
    return decorator
#装饰器decorator 返回wrapper
#装饰器生成函数返回装饰器 外层的生成函数可以接收参数
# func = time(2)(func) = decorator(func) = wrapper
@timer(2)
def func():
    time.sleep(3)

func()
addd = lambda *args: sum(args)
print(addd(1,2,3,4,5))