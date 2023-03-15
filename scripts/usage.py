def my_func(a, b):
    return 2*a + b

array_1 = np.arange(1, 5)
array_2 = np.arange(2, 7)
normal = my_func(array_1, array_2)
ours = my_func(log_array(array_1, array_2)
assert np.all(np.log(my_func(array_1, array_2)) == np.log(my_func, 
