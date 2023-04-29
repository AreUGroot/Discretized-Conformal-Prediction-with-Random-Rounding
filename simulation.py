import numpy as np
from dataclasses import dataclass


@dataclass
class MyData:
    n_samples: int
    p_features: int
    x_features: np.ndarray = np.empty(1)
    coefficients: np.ndarray = np.empty(1)
    y_labels: np.ndarray = np.empty(1)


def generate_features(my_data: MyData, method: int) -> MyData:
    n_samples, p_features = my_data.n_samples, my_data.p_features
    if method == 1:
        my_data.x_features = np.random.normal(size=(n_samples, p_features))
    elif method == 2:
        pass
    return my_data

def generate_labels(my_data: MyData, method: int, scale: float) -> MyData:
    n_samples, p_features = my_data.n_samples, my_data.p_features
    if method == 1:
        x_features = my_data.x_features
        # Specify the coefficients beta
        beta_coefficients = np.concatenate((np.ones(p_features - 3), np.ones(3)))
        # Compute conditional mean
        conditional_mean = x_features @ beta_coefficients 
        # Generate noises
        noises = np.random.normal(size=n_samples)
    elif method == 2:
        x_features = my_data.x_features
        # Specify the coefficients beta
        beta_coefficients = np.ones(p_features)
        # Compute conditional mean
        conditional_mean = x_features @ beta_coefficients 
        # Generate noises
        noises = np.random.normal(size=n_samples, scale=scale)
    # Get coefficients and labels
    my_data.coefficients = beta_coefficients
    my_data.y_labels = conditional_mean + noises
    return my_data

def sample_dataset(n_samples: int, p_features: int, 
                   method_features: int, method_labels: int,
                   noise_sigma: int) -> MyData:
    my_data = MyData(n_samples, p_features)
    my_data = generate_features(my_data, method_features)
    my_data = generate_labels(my_data, method_labels, noise_sigma)
    return my_data


#|%%--%%| <XAnc2fhNPd|UDUDsfacJt>


# from typing import NewType

# class PositiveInt(int):
    # def __new__(cls, value):
        # if value < 0:
            # raise ValueError("Value must be positive")
        # return super().__new__(cls, value)

# UserId = NewType('UserID', PositiveInt)

# a = PositiveInt(3)
# type(a)
# b = 3
# type(b)
# c = UserId(-123)
# type(c)


# class MyClass:
    # def __new__(cls):
        # print("__new__ called")
        # instance = super().__new__(cls)
        # return instance


# my_obj = MyClass()

# class MyClassTwo:
    # def __new__(cls):
        # print("__new__ called")
        # instance = super().__new__(cls)
        # return instance

    # def __init__(self):
        # print("__init__ called")
        # self.my_var = 42

# my_obj_two = MyClassTwo()



