import numpy as np_real  # Реальный numpy для структур и типов

class MyNumpyArray(np_real.ndarray):
    def __new__(cls, input_array):
        # Преобразуем входные данные в numpy.ndarray
        obj = np_real.asarray(input_array).view(cls)
        return obj

class MyNumpy:
    array = np_real.array  # Используем numpy.array напрямую
    zeros = np_real.zeros  # Используем numpy.zeros
    ones = np_real.ones    # Используем numpy.ones
    hstack = np_real.hstack
    unique = np_real.unique
    where = np_real.where
    exp = np_real.exp
    log = np_real.log
    clip = np_real.clip
    isnan = np_real.isnan  # Для проверки NaN, если потребуется